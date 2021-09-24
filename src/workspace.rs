use std::any::Any;
use std::cell::RefCell;
use std::thread::LocalKey;

/// A workspace that contains type-erased objects.
///
/// The workspace is intended to hold intermediate data used as workspace in computations.
/// It is optimized particularly for the case where the same type is accessed many times in a row.
///
/// TODO: Tests
#[derive(Debug, Default)]
pub struct Workspace {
    workspaces: Vec<Box<dyn Any>>,
}

impl Workspace {
    pub fn get_or_insert_with<W, F>(&mut self, create: F) -> &mut W
    where
        W: 'static,
        F: FnOnce() -> W,
    {
        // Note: We treat the Vec as a stack, so we search from the end of the vector.
        let existing_ws_idx = self.workspaces.iter().rposition(|ws| ws.is::<W>());
        let idx = match existing_ws_idx {
            Some(idx) => idx,
            None => {
                let w = create();
                let idx = self.workspaces.len();
                self.workspaces.push(Box::new(w) as Box<dyn Any>);
                idx
            }
        };

        // We heuristically assume that the same object is likely to be accessed
        // many times in sequence. Therefore we make sure that the object is the last entry,
        // so that on the next lookup, we'll immediately find the correct object
        let last = self.workspaces.len() - 1;
        self.workspaces.swap(idx, last);

        let entry = &mut self.workspaces[last];
        entry
            .downcast_mut()
            .expect("Internal error: Downcasting can by definition not fail")
    }

    pub fn get_or_default<W>(&mut self) -> &mut W
    where
        W: 'static + Default,
    {
        self.get_or_insert_with(Default::default)
    }
}

/// Runs the provided closure with the thread-local workspace as an argument.
///
/// This simplifies working with [`Workspace`] when it's stored as a thread-local variable.
///
/// Note that the typed workspace must have a [`Default`] implementation.
///
/// # Examples
///
/// ```rust
/// # use std::cell::RefCell;
/// # use fenris::workspace::{with_thread_local_workspace, Workspace};
/// thread_local! { static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default()); }
///
/// #[derive(Default)]
/// struct MyWorkspace {
///     buffer: Vec<usize>
/// }
///
/// fn main() {
///     let sum: usize = with_thread_local_workspace(&WORKSPACE, |ws: &mut MyWorkspace| {
///         // This is of course completely nonsense, we just show how you can easily use a thread-local workspace
///         // and produce a result which is returned.
///         ws.buffer.clear();
///         ws.buffer.extend_from_slice(&[1, 4, 3]);
///         ws.buffer.iter().sum()
///     });
///     println!("Sum = {}", sum);
/// }
/// ```
pub fn with_thread_local_workspace<W: 'static + Default, T>(
    workspace: &'static LocalKey<RefCell<Workspace>>,
    f: impl FnOnce(&mut W) -> T)
-> T {
    workspace.with(|refcell_ws| {
        let mut type_erased_workspace = refcell_ws.borrow_mut();
        let workspace = type_erased_workspace.get_or_default();
        f(workspace)
    })
}
