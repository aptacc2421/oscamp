//! Counters for **page-internal** allocation cost (alignment padding, gross block
//! size vs requested, splits). Enabled with crate feature `heap-profile`.

use core::sync::atomic::{AtomicU64, Ordering};

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_OK: AtomicU64 = AtomicU64::new(0);
static FREELIST_FAIL: AtomicU64 = AtomicU64::new(0);
static SUM_REQ: AtomicU64 = AtomicU64::new(0);
static SUM_PHY: AtomicU64 = AtomicU64::new(0);
static SUM_ALIGN_PRE_USER: AtomicU64 = AtomicU64::new(0);
static SPLIT_TAIL: AtomicU64 = AtomicU64::new(0);
static ABSORB_WHOLE_TAIL: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static COALESCE_LOOP_ITERS: AtomicU64 = AtomicU64::new(0);
static BIN_BLOCK_TRIES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct LabHeapProfileSnapshot {
    pub alloc_calls: u64,
    pub alloc_ok: u64,
    pub freelist_fail: u64,
    pub sum_requested: u64,
    pub sum_physical: u64,
    pub sum_align_pre_user: u64,
    pub split_tail: u64,
    pub absorb_whole_tail: u64,
    pub dealloc_calls: u64,
    pub coalesce_loop_iters: u64,
    pub bin_block_tries: u64,
}

impl LabHeapProfileSnapshot {
    /// Average gross block size (header + footer + alignment slack + tail) per successful alloc.
    pub fn avg_gross_per_ok(&self) -> f64 {
        if self.alloc_ok == 0 {
            return 0.0;
        }
        self.sum_physical as f64 / self.alloc_ok as f64
    }

    /// Average bytes of alignment padding before the user pointer (`user - base - hdr_reserved`).
    pub fn avg_align_pre_user(&self) -> f64 {
        if self.alloc_ok == 0 {
            return 0.0;
        }
        self.sum_align_pre_user as f64 / self.alloc_ok as f64
    }
}

pub fn snapshot() -> LabHeapProfileSnapshot {
    LabHeapProfileSnapshot {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_ok: ALLOC_OK.load(Ordering::Relaxed),
        freelist_fail: FREELIST_FAIL.load(Ordering::Relaxed),
        sum_requested: SUM_REQ.load(Ordering::Relaxed),
        sum_physical: SUM_PHY.load(Ordering::Relaxed),
        sum_align_pre_user: SUM_ALIGN_PRE_USER.load(Ordering::Relaxed),
        split_tail: SPLIT_TAIL.load(Ordering::Relaxed),
        absorb_whole_tail: ABSORB_WHOLE_TAIL.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        coalesce_loop_iters: COALESCE_LOOP_ITERS.load(Ordering::Relaxed),
        bin_block_tries: BIN_BLOCK_TRIES.load(Ordering::Relaxed),
    }
}

#[inline]
pub(crate) fn note_alloc_enter() {
    ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_freelist_fail() {
    FREELIST_FAIL.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_alloc_ok(
    req: usize,
    need_phy: usize,
    base: usize,
    user: usize,
    hdr_reserved: usize,
    split_tail: bool,
    absorb: bool,
    bin_tries: u32,
) {
    ALLOC_OK.fetch_add(1, Ordering::Relaxed);
    SUM_REQ.fetch_add(req as u64, Ordering::Relaxed);
    SUM_PHY.fetch_add(need_phy as u64, Ordering::Relaxed);
    let apu = user.saturating_sub(base).saturating_sub(hdr_reserved);
    SUM_ALIGN_PRE_USER.fetch_add(apu as u64, Ordering::Relaxed);
    if split_tail {
        SPLIT_TAIL.fetch_add(1, Ordering::Relaxed);
    }
    if absorb {
        ABSORB_WHOLE_TAIL.fetch_add(1, Ordering::Relaxed);
    }
    BIN_BLOCK_TRIES.fetch_add(bin_tries as u64, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_dealloc() {
    DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn note_coalesce_iters(iters: u64) {
    COALESCE_LOOP_ITERS.fetch_add(iters, Ordering::Relaxed);
}
