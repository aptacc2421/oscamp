//! Default [`BuildHasher`](core::hash::BuildHasher) for [`crate::collections::HashMap`] /
//! [`crate::collections::HashSet`], seeded from [`arceos_api::random::ax_random_u128`].

use core::hash::BuildHasher;
use foldhash::fast::FoldHasher;
use foldhash::SharedSeed;

/// BuildHasher using foldhash, with global seed material from the kernel PRNG.
#[derive(Clone, Debug)]
pub struct AxRandomState {
    per_hasher_seed: u64,
    shared: SharedSeed,
}

impl Default for AxRandomState {
    fn default() -> Self {
        let r = arceos_api::random::ax_random_u128();
        let lo = r as u64;
        let hi = (r >> 64) as u64;
        let mut seed = lo ^ hi;
        if seed == 0 {
            seed = 0x9e37_79b9_7f4a_7c15;
        }
        Self {
            per_hasher_seed: lo.rotate_left(17) ^ hi,
            shared: SharedSeed::from_u64(seed),
        }
    }
}

impl BuildHasher for AxRandomState {
    type Hasher = FoldHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        FoldHasher::with_seed(self.per_hasher_seed, &self.shared)
    }
}
