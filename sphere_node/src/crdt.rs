//! Last-write-wins CRDT for neural network weight synchronisation.
//!
//! Each weight is wrapped in a [`WeightCell`] that pairs the value with a
//! logical clock (Lamport timestamp).  Merging two `WeightCell`s always keeps
//! the value with the higher clock, making the operation commutative,
//! associative and idempotent — the three CRDT requirements.
//!
//! A [`WeightSet`] is an ordered collection of `WeightCell`s representing one
//! complete layer or the full parameter vector of a node.

use serde::{Deserialize, Serialize};

/// A single weight with an associated Lamport clock.
///
/// `merge` implements the LWW-Element-Set register: the cell with the greater
/// `clock` wins; on ties the higher `value` wins to break symmetry
/// deterministically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightCell {
    /// The floating-point weight value.
    pub value: f64,
    /// Lamport clock — strictly monotonically increasing at write time.
    pub clock: u64,
    /// Originating node identifier (used for tie-breaking).
    pub origin: u64,
}

impl WeightCell {
    /// Create a new cell with clock = 0.
    pub fn new(value: f64, origin: u64) -> Self {
        Self { value, clock: 0, origin }
    }

    /// Increment the clock and set a new value (local write).
    pub fn write(&mut self, value: f64) {
        self.clock += 1;
        self.value = value;
    }

    /// Merge `other` into `self`, keeping the causally-later write.
    ///
    /// Returns `true` when `self` was updated.
    pub fn merge(&mut self, other: &WeightCell) -> bool {
        if other.clock > self.clock
            || (other.clock == self.clock && other.value > self.value)
        {
            self.value = other.value;
            self.clock = other.clock;
            self.origin = other.origin;
            true
        } else {
            false
        }
    }
}

/// An ordered vector of [`WeightCell`]s representing all parameters for one
/// node (or one layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSet {
    pub cells: Vec<WeightCell>,
}

impl WeightSet {
    /// Allocate a new weight set of `size` cells, all initialised to 0 with
    /// origin node `node_id`.
    pub fn new(size: usize, node_id: u64) -> Self {
        Self {
            cells: (0..size).map(|_| WeightCell::new(0.0, node_id)).collect(),
        }
    }

    /// Write a slice of values into the set, advancing each cell's clock.
    ///
    /// # Panics
    /// Panics if `values.len() != self.cells.len()`.
    pub fn write_all(&mut self, values: &[f64]) {
        assert_eq!(
            values.len(),
            self.cells.len(),
            "WeightSet::write_all: value slice length mismatch"
        );
        for (cell, &v) in self.cells.iter_mut().zip(values) {
            cell.write(v);
        }
    }

    /// Cell-wise merge with `other`.  Both sets must have the same length.
    ///
    /// Returns the number of cells that were updated.
    pub fn merge(&mut self, other: &WeightSet) -> usize {
        assert_eq!(
            self.cells.len(),
            other.cells.len(),
            "WeightSet::merge: length mismatch"
        );
        let mut count = 0usize;
        for (self_cell, other_cell) in self.cells.iter_mut().zip(other.cells.iter()) {
            if self_cell.merge(other_cell) {
                count += 1;
            }
        }
        count
    }

    /// Return the current weight values as a plain vector.
    pub fn values(&self) -> Vec<f64> {
        self.cells.iter().map(|c| c.value).collect()
    }

    /// Maximum logical clock across all cells — used to assess freshness.
    pub fn max_clock(&self) -> u64 {
        self.cells.iter().map(|c| c.clock).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lww_merge_higher_clock_wins() {
        let mut a = WeightCell { value: 1.0, clock: 3, origin: 1 };
        let b = WeightCell { value: 2.0, clock: 5, origin: 2 };
        let updated = a.merge(&b);
        assert!(updated);
        assert_eq!(a.value, 2.0);
        assert_eq!(a.clock, 5);
    }

    #[test]
    fn lww_merge_lower_clock_ignored() {
        let mut a = WeightCell { value: 1.0, clock: 5, origin: 1 };
        let b = WeightCell { value: 9.0, clock: 2, origin: 2 };
        let updated = a.merge(&b);
        assert!(!updated);
        assert_eq!(a.value, 1.0);
    }

    #[test]
    fn weight_set_merge_counts_updates() {
        let mut ws_a = WeightSet::new(4, 1);
        ws_a.write_all(&[1.0, 2.0, 3.0, 4.0]);

        let mut ws_b = WeightSet::new(4, 2);
        // Give b a later write on two of the cells.
        ws_b.cells[1].write(20.0);
        ws_b.cells[1].write(21.0); // clock = 2
        ws_b.cells[3].write(40.0); // clock = 1

        // ws_a has clock=1 on all cells; ws_b beats it on cells 1 (clock 2)
        // and 3 (clock 1, value 40 > 4).
        let updates = ws_a.merge(&ws_b);
        assert_eq!(updates, 2);
        assert_eq!(ws_a.cells[1].value, 21.0);
        assert_eq!(ws_a.cells[3].value, 40.0);
    }
}
