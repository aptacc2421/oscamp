use alloc::vec::Vec;
use alloc::sync::Arc;
use axerrno::AxResult;
use memory_addr::VirtAddr;
use axhal::paging::MappingFlags;
use axmm::AddrSpace;

pub struct VmDev {
    start: VirtAddr,
    size: usize,
}

impl VmDev {
    pub fn new(start: VirtAddr, size: usize) -> Self {
        Self { start, size }
    }

    pub fn handle_mmio(&self, addr: VirtAddr , aspace: &mut AddrSpace) -> AxResult {
        let mapping_flags = MappingFlags::from_bits(0xf).unwrap();
        // Passthrough-Mode
        aspace.map_linear(addr, addr.as_usize().into(), 4096, mapping_flags)
    }

    pub fn check_addr(&self, addr: VirtAddr) -> bool {
        addr >= self.start && addr < (self.start + self.size)
    }
}

pub struct VmDevGroup {
    devices: Vec<Arc<VmDev>>
}

impl VmDevGroup {
    pub fn new() -> Self {
        Self { devices: Vec::new() }
    }

    pub fn add_dev(&mut self, addr: VirtAddr, size: usize) {
        let dev = VmDev::new(addr, size);
        self.devices.push(Arc::new(dev));
    }

    pub fn find_dev(&self, addr: VirtAddr) -> Option<Arc<VmDev>> {
        self.devices
            .iter()
            .find(|&dev| dev.check_addr(addr))
            .cloned()
    }
}
