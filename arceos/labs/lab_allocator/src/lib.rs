//! Lab byte allocator: **segregated free lists** (by `floor_log2` of block size) +
//! **footers** for O(1) physical coalesce on free. No `rlsf`.
//!
//! Route B step 2: allocation scans bins from `bin(need_est)` upward; each bin is
//! an intrusive singly linked list **sorted by block address within each bin**.
//! Used/free blocks carry a
//! **footer** at `base + phy - 8` so backward merge does not need a global
//! address-sorted chain.

#![no_std]

#[cfg(feature = "heap-profile")]
mod profile;

#[cfg(feature = "heap-profile")]
pub use profile::{snapshot as heap_profile_snapshot, LabHeapProfileSnapshot};

use allocator::{AllocError, AllocResult, BaseAllocator, ByteAllocator};
use core::alloc::Layout;
use core::ptr::NonNull;

const MIN_BLOCK: usize = 32;
const HDR_RESERVED: usize = core::mem::size_of::<usize>() * 2;
/// `floor_log2(size)` for `size >= MIN_BLOCK` maps to bin `lg - BIN_SHIFT`.
const BIN_SHIFT: usize = 5; // bins start at sizes >= 32 (MIN_BLOCK)
const NUM_BINS: usize = 40; // up to ~16 TiB class

#[repr(C)]
struct FreeNode {
    /// Physical size, **bit0 = 0** when free.
    size: usize,
    next_bin: Option<NonNull<FreeNode>>,
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
fn phys(sz: usize) -> usize {
    sz & !1usize
}

#[inline]
fn is_free_hdr(w: usize) -> bool {
    w & 1 == 0
}

/// `floor_log2` for `sz >= 1`.
#[inline]
fn floor_log2(sz: usize) -> usize {
    (usize::BITS as usize) - sz.leading_zeros() as usize - 1
}

#[inline]
fn bin_index_for_phy(sz: usize) -> usize {
    let sz = sz.max(MIN_BLOCK);
    let lg = floor_log2(sz);
    lg.saturating_sub(BIN_SHIFT).min(NUM_BINS - 1)
}

/// Conservative first bin to search for `layout`.
#[inline]
fn bin_index_for_layout(layout: &Layout) -> usize {
    let est = layout
        .size()
        .saturating_add(HDR_RESERVED)
        .saturating_add(layout.align())
        .max(MIN_BLOCK);
    bin_index_for_phy(est)
}

#[inline]
unsafe fn write_footer(base: usize, phy: usize, word: usize) {
    *((base + phy - core::mem::size_of::<usize>()) as *mut usize) = word;
}

pub struct LabByteAllocator {
    regions: [(usize, usize); 32],
    region_count: usize,
    /// Lowest byte in any registered region (guards `base-8` on backward merge).
    heap_low: usize,
    /// Highest byte exclusive (for validating `next` physical merge).
    heap_end: usize,
    free_bins: [Option<NonNull<FreeNode>>; NUM_BINS],
    total_bytes: usize,
    used_bytes: usize,
}

unsafe impl Send for LabByteAllocator {}
unsafe impl Sync for LabByteAllocator {}

impl LabByteAllocator {
    pub const fn new() -> Self {
        Self {
            regions: [(0, 0); 32],
            region_count: 0,
            heap_low: usize::MAX,
            heap_end: 0,
            free_bins: [None; NUM_BINS],
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

    /// Remove `node` from `free_bins[bi]` singly linked list.
    unsafe fn unlink_bin(
        bins: &mut [Option<NonNull<FreeNode>>; NUM_BINS],
        bi: usize,
        node: NonNull<FreeNode>,
    ) {
        let want = node_addr(node);
        let Some(mut head) = bins[bi] else {
            debug_assert!(false, "unlink_bin: empty bin");
            return;
        };
        if node_addr(head) == want {
            bins[bi] = head.as_ref().next_bin;
            head.as_mut().next_bin = None;
            return;
        }
        let mut prev = head;
        while let Some(mut nxt) = prev.as_ref().next_bin {
            if node_addr(nxt) == want {
                prev.as_mut().next_bin = nxt.as_ref().next_bin;
                nxt.as_mut().next_bin = None;
                return;
            }
            prev = nxt;
        }
        debug_assert!(false, "unlink_bin: node not in bin");
    }

    /// Insert `node` into `free_bins[bi]` in **increasing address** order.
    unsafe fn link_bin_addr_sorted(
        bins: &mut [Option<NonNull<FreeNode>>; NUM_BINS],
        bi: usize,
        mut node: NonNull<FreeNode>,
    ) {
        let na = node_addr(node);
        let mut cur = bins[bi];
        let mut prev_slot: *mut Option<NonNull<FreeNode>> = &mut bins[bi];
        while let Some(c) = cur {
            if node_addr(c) >= na {
                break;
            }
            prev_slot = &mut (*c.as_ptr()).next_bin;
            cur = c.as_ref().next_bin;
        }
        node.as_mut().next_bin = cur;
        *prev_slot = Some(node);
    }

    /// Walk bin `bi` to unlink block at `base` (used after coalesce changed address).
    unsafe fn unlink_bin_by_addr(
        bins: &mut [Option<NonNull<FreeNode>>; NUM_BINS],
        bi: usize,
        base: usize,
    ) -> Option<NonNull<FreeNode>> {
        let mut cur = bins[bi];
        while let Some(n) = cur {
            if node_addr(n) == base {
                Self::unlink_bin(bins, bi, n);
                return Some(n);
            }
            cur = n.as_ref().next_bin;
        }
        None
    }

    /// Try merge with previous physical block using footer at `base-8`.
    unsafe fn try_merge_prev(
        bins: &mut [Option<NonNull<FreeNode>>; NUM_BINS],
        heap_low: usize,
        base: &mut usize,
        phy: &mut usize,
    ) {
        if *base < core::mem::size_of::<usize>() || *base <= heap_low {
            return;
        }
        let prev_f = *(((*base).saturating_sub(core::mem::size_of::<usize>())) as *const usize);
        if !is_free_hdr(prev_f) {
            return;
        }
        let psz = phys(prev_f);
        let pstart = (*base).saturating_sub(psz);
        if pstart < heap_low || pstart + psz != *base {
            return;
        }
        let bi = bin_index_for_phy(psz);
        if Self::unlink_bin_by_addr(bins, bi, pstart).is_none() {
            return;
        }
        *phy += psz;
        *base = pstart;
    }

    /// Try merge with next physical block at `base + phy`.
    unsafe fn try_merge_next(
        heap_end: usize,
        bins: &mut [Option<NonNull<FreeNode>>; NUM_BINS],
        base: usize,
        phy: &mut usize,
    ) {
        let nstart = base + *phy;
        if nstart >= heap_end {
            return;
        }
        let nw = *(nstart as *const usize);
        if !is_free_hdr(nw) {
            return;
        }
        let nsz = phys(nw);
        let bi = bin_index_for_phy(nsz);
        if Self::unlink_bin_by_addr(bins, bi, nstart).is_none() {
            return;
        }
        *phy += nsz;
    }

    /// Free `phy` bytes at `base`, coalesce, write header+footer, insert into bin.
    unsafe fn insert_free_coalesced(&mut self, mut base: usize, mut phy: usize) {
        let heap_low = self.heap_low;
        let heap_end = self.heap_end;
        let bins = &mut self.free_bins;
        let mut co_iters = 0u64;
        loop {
            co_iters += 1;
            let ob = base;
            let op = phy;
            Self::try_merge_prev(bins, heap_low, &mut base, &mut phy);
            Self::try_merge_next(heap_end, bins, base, &mut phy);
            if base == ob && phy == op {
                break;
            }
        }
        #[cfg(feature = "heap-profile")]
        profile::note_coalesce_iters(co_iters);
        #[cfg(not(feature = "heap-profile"))]
        let _ = co_iters;

        let node = base as *mut FreeNode;
        (*node).size = phy & !1;
        (*node).next_bin = None;
        write_footer(base, phy, phy & !1);

        let bi = bin_index_for_phy(phy);
        Self::link_bin_addr_sorted(&mut self.free_bins, bi, NonNull::new_unchecked(node));
    }

    unsafe fn try_alloc_from_block(
        &mut self,
        free: NonNull<FreeNode>,
        bi: usize,
        layout: Layout,
        #[cfg_attr(not(feature = "heap-profile"), allow(unused_variables))] bin_block_tries: u32,
    ) -> AllocResult<NonNull<u8>> {
        let align = layout.align().max(core::mem::size_of::<usize>());
        let req = layout.size();
        let base = node_addr(free);
        let fsize = phys(free.as_ref().size);
        let fend = base + fsize;

        let user = align_up(base + HDR_RESERVED, align);
        if user < base + HDR_RESERVED {
            return Err(AllocError::NoMemory);
        }

        let mut need_phy = user
            .checked_sub(base)
            .and_then(|h| h.checked_add(req))
            .ok_or(AllocError::NoMemory)?;
        need_phy = align_up(need_phy, core::mem::size_of::<usize>()).max(MIN_BLOCK);

        if base + need_phy > fend {
            return Err(AllocError::NoMemory);
        }

        let rem_before_absorb = fend - (base + need_phy);
        let absorb = rem_before_absorb > 0 && rem_before_absorb < MIN_BLOCK;
        if absorb {
            need_phy = fend - base;
        }

        Self::unlink_bin(&mut self.free_bins, bi, free);

        let rem_after = fend - (base + need_phy);
        let split_tail = rem_after >= MIN_BLOCK;
        if split_tail {
            let tail = base + need_phy;
            let tp = tail as *mut FreeNode;
            (*tp).size = rem_after & !1;
            (*tp).next_bin = None;
            write_footer(tail, rem_after, rem_after & !1);
            self.insert_free_coalesced(tail, rem_after);
        }

        let user_nn = NonNull::new_unchecked(user as *mut u8);
        let back_off = user - base;
        if back_off < core::mem::size_of::<usize>() {
            return Err(AllocError::InvalidParam);
        }
        *(base as *mut usize) = need_phy | 1;
        *((user as *mut usize).offset(-1)) = back_off;
        write_footer(base, need_phy, need_phy | 1);

        self.used_bytes = self.used_bytes.saturating_add(req);
        #[cfg(feature = "heap-profile")]
        profile::note_alloc_ok(
            req,
            need_phy,
            base,
            user,
            HDR_RESERVED,
            split_tail,
            absorb,
            bin_block_tries,
        );
        Ok(user_nn)
    }

    unsafe fn alloc_from_bins(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        #[cfg(feature = "heap-profile")]
        profile::note_alloc_enter();
        let start_bin = bin_index_for_layout(&layout);
        let mut block_tries = 0u32;
        for bi in start_bin..NUM_BINS {
            let mut cur = self.free_bins[bi];
            while let Some(free) = cur {
                block_tries = block_tries.saturating_add(1);
                let next = free.as_ref().next_bin;
                match self.try_alloc_from_block(free, bi, layout, block_tries) {
                    Ok(p) => return Ok(p),
                    Err(AllocError::NoMemory) => cur = next,
                    Err(e) => return Err(e),
                }
            }
        }
        #[cfg(feature = "heap-profile")]
        profile::note_freelist_fail();
        Err(AllocError::NoMemory)
    }

    unsafe fn free_block(&mut self, user: NonNull<u8>, layout: Layout) {
        #[cfg(feature = "heap-profile")]
        profile::note_dealloc();
        let p = user.as_ptr() as usize;
        let back_off = *((p as *const usize).offset(-1));
        let base = p - back_off;
        let hdr_phy = phys(*(base as *const usize));
        debug_assert!(p + layout.size() <= base + hdr_phy);

        let node = base as *mut FreeNode;
        (*node).size = hdr_phy & !1;
        (*node).next_bin = None;
        write_footer(base, hdr_phy, hdr_phy & !1);

        self.insert_free_coalesced(base, hdr_phy);
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
        self.heap_low = start;
        self.heap_end = start.saturating_add(size);
        self.push_region(start, size).expect("region table");
        unsafe {
            let n = start as *mut FreeNode;
            (*n).size = size & !1;
            (*n).next_bin = None;
            write_footer(start, size, size & !1);
            let bi = bin_index_for_phy(size);
            Self::link_bin_addr_sorted(&mut self.free_bins, bi, NonNull::new_unchecked(n));
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
        self.heap_low = self.heap_low.min(start);
        self.heap_end = self.heap_end.max(start.saturating_add(size));
        self.push_region(start, size)?;
        unsafe {
            let n = start as *mut FreeNode;
            (*n).size = size & !1;
            (*n).next_bin = None;
            write_footer(start, size, size & !1);
            self.insert_free_coalesced(start, size);
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
        unsafe { self.alloc_from_bins(layout) }
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
