//! Counters and one-shot logging for the **global heap expand path** (byte alloc miss
//! → `alloc_pages` → `add_memory`). Used with feature `heap-profile`.

use crate::{DefaultByteAllocator, GlobalAllocator};
use allocator::{AllocError, ByteAllocator, PageAllocator};
use core::alloc::Layout;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static EXPAND_OK: AtomicU64 = AtomicU64::new(0);
static EXPAND_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_EXPAND_SIZE: AtomicUsize = AtomicUsize::new(0);
static LAST_OLD_BYTE_TOTAL: AtomicUsize = AtomicUsize::new(0);
static LAST_FAIL_LAYOUT_SIZE: AtomicUsize = AtomicUsize::new(0);
static LAST_FAIL_LAYOUT_ALIGN: AtomicUsize = AtomicUsize::new(0);
static LAST_FAIL_EXPAND_TRY: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct ExpandHeapProfileSnapshot {
    pub expand_ok: u64,
    pub expand_bytes_total: u64,
    pub last_expand_size: usize,
    pub last_old_byte_total: usize,
    pub last_fail_layout_size: usize,
    pub last_fail_layout_align: usize,
    pub last_fail_expand_try: usize,
}

pub fn expand_snapshot() -> ExpandHeapProfileSnapshot {
    ExpandHeapProfileSnapshot {
        expand_ok: EXPAND_OK.load(Ordering::Relaxed),
        expand_bytes_total: EXPAND_BYTES.load(Ordering::Relaxed),
        last_expand_size: LAST_EXPAND_SIZE.load(Ordering::Relaxed),
        last_old_byte_total: LAST_OLD_BYTE_TOTAL.load(Ordering::Relaxed),
        last_fail_layout_size: LAST_FAIL_LAYOUT_SIZE.load(Ordering::Relaxed),
        last_fail_layout_align: LAST_FAIL_LAYOUT_ALIGN.load(Ordering::Relaxed),
        last_fail_expand_try: LAST_FAIL_EXPAND_TRY.load(Ordering::Relaxed),
    }
}

pub fn on_expand_success(expand_size: usize, old_byte_total: usize) {
    EXPAND_OK.fetch_add(1, Ordering::Relaxed);
    EXPAND_BYTES.fetch_add(expand_size as u64, Ordering::Relaxed);
    LAST_EXPAND_SIZE.store(expand_size, Ordering::Relaxed);
    LAST_OLD_BYTE_TOTAL.store(old_byte_total, Ordering::Relaxed);
    trace!(
        target: "heap_profile",
        "heap expand ok: +{:#x} bytes (byte_total was {:#x})",
        expand_size,
        old_byte_total
    );
}

pub fn on_page_alloc_failed(
    ga: &GlobalAllocator,
    balloc: &DefaultByteAllocator,
    layout: &Layout,
    old_byte_total: usize,
    expand_size: usize,
    page_err: &AllocError,
) {
    LAST_FAIL_LAYOUT_SIZE.store(layout.size(), Ordering::Relaxed);
    LAST_FAIL_LAYOUT_ALIGN.store(layout.align(), Ordering::Relaxed);
    LAST_FAIL_EXPAND_TRY.store(expand_size, Ordering::Relaxed);

    let (page_used, page_avail) = {
        let p = ga.palloc.lock();
        (p.used_pages(), p.available_pages())
    };

    let lab = lab_allocator::heap_profile_snapshot();

    warn!(
        target: "heap_profile",
        "heap exhausted: page_alloc_err={:?} layout(size={} align={}) expand_try={:#x} \
         byte_heap total={:#x} used={:#x} avail={:#x} pages_used={} pages_avail={} \
         expand_ok={} expand_bytes_total={:#x} lab={:?}",
        page_err,
        layout.size(),
        layout.align(),
        expand_size,
        old_byte_total,
        balloc.used_bytes(),
        balloc.available_bytes(),
        page_used,
        page_avail,
        EXPAND_OK.load(Ordering::Relaxed),
        EXPAND_BYTES.load(Ordering::Relaxed),
        lab
    );
}
