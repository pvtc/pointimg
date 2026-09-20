//! Formatage humain et garde des générations de worker.
//!
//! Les implémentations vivent dans `pointimg::frontend`, partagées avec le
//! frontend GTK4 ; ce module ne fait que les ré-exposer en `pub(crate)`.

pub(crate) use pointimg::frontend::{
    describe_parameter_change, format_duration, format_memory, generation_is_current,
};
