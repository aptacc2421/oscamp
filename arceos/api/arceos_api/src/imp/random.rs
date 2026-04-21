//! Platform pseudo-random values; delegates to `axhal::misc::random`.

#[inline]
pub fn ax_random_u128() -> u128 {
    axhal::misc::random()
}
