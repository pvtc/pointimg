//! Human formatting and worker generation guard.
//!
//! The implementations live in `pointimg::frontend`, shared with the GTK4
//! frontend; this module only re-exposes them as `pub(crate)`.

pub(crate) use pointimg::frontend::{
    describe_parameter_change, format_duration, generation_is_current,
};
