//! Lab byte allocator: free list sorted by address, coalesce adjacent on insert,
//! split on alloc. No `rlsf` dependency.

#![no_std]

use allocator::{AllocError, AllocResult, BaseAllocator, ByteAllocator};
use core::alloc::Layout;
use core::ptr::NonNull;

/// Minimum physical block (must hold `FreeNode` + footer slack).
const MIN_BLOCK: usize = 32;
/// Bytes reserved at block start before alignment padding / user (`phy` + padding).
const HDR_RESERVED: usize = core::mem::size_of::<usize>() * 2;

#[repr(C)]
struct FreeNode {
    /// Physical size, **bit0 = 0** when free.
    size: usize,
    next: Option<NonNull<FreeNode>>,
}

#[inline]
fn node_addr(n: NonNull<FreeNode>) -> usize {
    n.as_ptr() as usize
}

#[inline]
fn align_up(p: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (p + align - 1) & !(align - 1)
}

#[inline]
fn phy(sz: usize) -> usize {
    sz & !1usize
}

pub struct LabByteAllocator {
    regions: [(usize, usize); 32],
    region_count: usize,
    free_head: Option<NonNull<FreeNode>>,
    total_bytes: usize,
    used_bytes: usize,
}

// `NonNull` is not `Send`/`Sync` by default; the global allocator wraps this type
// in a spin lock and all pointers refer to the kernel heap region.
unsafe impl Send for LabByteAllocator {}
unsafe impl Sync for LabByteAllocator {}

impl LabByteAllocator {
    pub const fn new() -> Self {
        Self {
            regions: [(0, 0); 32],
            region_count: 0,
            free_head: None,
            total_bytes: 0,
            used_bytes: 0,
        }
    }

    fn overlaps_region(&self, start: usize, len: usize) -> bool {
        let end = start.saturating_add(len);
        for i in 0..self.region_count {
            let (rs, rl) = self.regions[i];
            let re = rs.saturating_add(rl);
            if start < re && end > rs {
                return true;
            }
        }
        false
    }

    fn push_region(&mut self, start: usize, len: usize) -> AllocResult {
        if self.region_count >= self.regions.len() {
            return Err(AllocError::InvalidParam);
        }
        self.regions[self.region_count] = (start, len);
        self.region_count += 1;
        Ok(())
    }

    /// Insert free block sorted by address, then merge physically adjacent neighbours.
    unsafe fn insert_free(&mut self, node: NonNull<FreeNode>) {
        let na = node_addr(node);

        let mut prev_field: *mut Option<NonNull<FreeNode>> = &mut self.free_head;
        let mut cur = self.free_head;

        loop {
            match cur {
                None => {
                    *prev_field = Some(node);
                    (*node.as_ptr()).next = None;
                    break;
                }
                Some(c) if node_addr(c) > na => {
                    (*node.as_ptr()).next = Some(c);
                    *prev_field = Some(node);
                    break;
                }
                Some(c) => {
                    prev_field = &mut (*c.as_ptr()).next;
                    cur = c.as_ref().next;
                }
            }
        }

        Self::coalesce_forward(&mut self.free_head);
    }

    /// Merge each node with following physically contiguous free blocks.
    unsafe fn coalesce_forward(head: &mut Option<NonNull<FreeNode>>) {
        loop {
            let mut changed = false;
            let mut cur_opt = *head;
            while let Some(mut c) = cur_opt {
                while let Some(nx) = c.as_ref().next {
                    let cend = node_addr(c) + phy(c.as_ref().size);
                    if node_addr(nx) != cend {
                        break;
                    }
                    let cs = phy(c.as_ref().size);
                    let ns = phy(nx.as_ref().size);
                    c.as_mut().size = (cs + ns) & !1;
                    c.as_mut().next = nx.as_ref().next;
                    changed = true;
                }
                cur_opt = c.as_ref().next;
            }
            if !changed {
                break;
            }
        }
    }

    unsafe fn alloc_from_freelist(
        &mut self,
        layout: Layout,
    ) -> AllocResult<NonNull<u8>> {
        let align = layout.align().max(core::mem::size_of::<usize>());
        let req = layout.size();

        let mut prev_field: *mut Option<NonNull<FreeNode>> = &mut self.free_head;
        let mut cur_opt = self.free_head;

        while let Some(free) = cur_opt {
            let base = node_addr(free);
            let fsize = phy(free.as_ref().size);
            let fend = base + fsize;

            let user = align_up(base + HDR_RESERVED, align);
            if user < base + HDR_RESERVED {
                prev_field = &mut (*free.as_ptr()).next;
                cur_opt = *prev_field;
                continue;
            }

            let need_phy = user
                .checked_sub(base)
                .and_then(|h| h.checked_add(req))
                .ok_or(AllocError::InvalidParam)?;
            let need_phy = align_up(need_phy, core::mem::size_of::<usize>()).max(MIN_BLOCK);

            if base + need_phy > fend {
                prev_field = &mut (*free.as_ptr()).next;
                cur_opt = *prev_field;
                continue;
            }

            let rem = fend - (base + need_phy);
            let next_in_list = free.as_ref().next;
            *prev_field = next_in_list;

            if rem >= MIN_BLOCK {
                let tail = base + need_phy;
                let tp = tail as *mut FreeNode;
                (*tp).size = rem & !1;
                (*tp).next = None;
                self.insert_free(NonNull::new_unchecked(tp));
            }

            let user_nn = NonNull::new_unchecked(user as *mut u8);
            let back_off = user - base;
            if back_off < core::mem::size_of::<usize>() {
                return Err(AllocError::InvalidParam);
            }
            *(base as *mut usize) = need_phy | 1;
            *((user as *mut usize).offset(-1)) = back_off;

            self.used_bytes = self.used_bytes.saturating_add(req);
            return Ok(user_nn);
        }

        Err(AllocError::NoMemory)
    }

    unsafe fn free_block(&mut self, user: NonNull<u8>, layout: Layout) {
        let p = user.as_ptr() as usize;
        let back_off = *((p as *const usize).offset(-1));
        let base = p - back_off;
        let hdr_phy = phy(*(base as *const usize));
        let end = p + layout.size();
        debug_assert_eq!(end - base, hdr_phy);

        let node = base as *mut FreeNode;
        (*node).size = hdr_phy & !1;
        (*node).next = None;
        self.insert_free(NonNull::new_unchecked(node));
        self.used_bytes = self.used_bytes.saturating_sub(layout.size());
    }
}

impl BaseAllocator for LabByteAllocator {
    fn init(&mut self, start: usize, size: usize) {
        assert!(self.region_count == 0);
        assert!(size >= MIN_BLOCK);
        assert!(start % core::mem::size_of::<usize>() == 0);
        *self = Self::new();
        self.total_bytes = size;
        self.push_region(start, size).expect("region table");
        unsafe {
            let n = start as *mut FreeNode;
            (*n).size = size & !1;
            (*n).next = None;
            self.free_head = Some(NonNull::new_unchecked(n));
        }
    }

    fn add_memory(&mut self, start: usize, size: usize) -> AllocResult {
        if size < MIN_BLOCK {
            return Err(AllocError::InvalidParam);
        }
        if start % core::mem::size_of::<usize>() != 0 {
            return Err(AllocError::InvalidParam);
        }
        if self.overlaps_region(start, size) {
            return Err(AllocError::MemoryOverlap);
        }
        self.total_bytes = self.total_bytes.saturating_add(size);
        self.push_region(start, size)?;
        unsafe {
            let n = start as *mut FreeNode;
            (*n).size = size & !1;
            (*n).next = None;
            self.insert_free(NonNull::new_unchecked(n));
        }
        Ok(())
    }
}

impl ByteAllocator for LabByteAllocator {
    fn alloc(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        if layout.size() == 0 {
            return Err(AllocError::InvalidParam);
        }
        if !layout.align().is_power_of_two() || layout.align() == 0 {
            return Err(AllocError::InvalidParam);
        }
        unsafe { self.alloc_from_freelist(layout) }
    }

    fn dealloc(&mut self, pos: NonNull<u8>, layout: Layout) {
        unsafe { self.free_block(pos, layout) }
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    fn available_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
}
