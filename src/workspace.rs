use std::any::Any;

/// A workspace that contains type-erased objects.
///
/// The workspace is intended to hold intermediate data used as workspace in computations.
/// It is optimized particularly for the case where the same type is accessed many times in a row.
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
                self.workspaces.push(Box::new(w) as Box<dyn Any>);
                self.workspaces.len()
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
