use alloc::sync::Arc;

use axmm::AddrSpace;
use axsync::Mutex;
use crate::vcpu::VmCpuRegisters;

/// Task extended data for the monolithic kernel.
pub struct TaskExt {
    /// The vcpu.
    pub vcpu: VmCpuRegisters,
    /// The virtual memory address space.
    pub aspace: Arc<Mutex<AddrSpace>>,
}

impl TaskExt {
    pub const fn new(vcpu: VmCpuRegisters, aspace: Arc<Mutex<AddrSpace>>) -> Self {
        Self {
            vcpu,
            aspace,
        }
    }
}

axtask::def_task_ext!(TaskExt);
