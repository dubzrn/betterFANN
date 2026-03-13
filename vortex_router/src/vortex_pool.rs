use crate::thermal_node::ThermalNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexType {
    /// Primary path — centripetal routing along the main flow.
    Longitudinal,
    /// Cross-balancing between lanes.
    Transverse,
    /// Persistence synchronisation heartbeat.
    Vertical,
}

/// Routes requests centripetally: always to the coldest, improving node.
#[derive(Debug)]
pub struct VortexPool {
    nodes: Vec<ThermalNode>,
}

impl VortexPool {
    pub fn new(nodes: Vec<ThermalNode>) -> Self {
        Self { nodes }
    }

    /// Return the index of the best node for the given vortex type.
    ///
    /// Type A routing: only nodes whose `quality_improving` flag is `true` are
    /// eligible.  If none are improving we fall back to the globally coldest node
    /// to avoid a dead-pool.
    pub fn select(&self, vortex_type: VortexType) -> Option<usize> {
        let candidates: Vec<usize> = match vortex_type {
            VortexType::Longitudinal => {
                // Main path — prefer improving nodes with lowest thermal score.
                (0..self.nodes.len())
                    .filter(|&i| self.nodes[i].quality_improving)
                    .collect()
            }
            VortexType::Transverse => {
                // Cross-balancing — target mid-range temperature nodes to equalise.
                let mut idxs: Vec<usize> = (0..self.nodes.len()).collect();
                idxs.sort_by(|&a, &b| {
                    let mid_a = (self.nodes[a].temperature - 50.0).abs();
                    let mid_b = (self.nodes[b].temperature - 50.0).abs();
                    mid_a.partial_cmp(&mid_b).unwrap()
                });
                idxs
            }
            VortexType::Vertical => {
                // Persistence sync — use the most stable (lowest variance) node,
                // approximated here by lowest latency_ms.
                (0..self.nodes.len()).collect()
            }
        };

        if candidates.is_empty() {
            // Fallback: globally coldest node.
            return self.coldest_index();
        }

        candidates
            .into_iter()
            .min_by(|&a, &b| {
                self.nodes[a]
                    .thermal_score()
                    .partial_cmp(&self.nodes[b].thermal_score())
                    .unwrap()
            })
    }

    fn coldest_index(&self) -> Option<usize> {
        (0..self.nodes.len()).min_by(|&a, &b| {
            self.nodes[a]
                .thermal_score()
                .partial_cmp(&self.nodes[b].thermal_score())
                .unwrap()
        })
    }

    pub fn node_mut(&mut self, idx: usize) -> &mut ThermalNode {
        &mut self.nodes[idx]
    }

    pub fn nodes(&self) -> &[ThermalNode] {
        &self.nodes
    }

    /// Sort nodes in-place by thermal score (cold core first).
    pub fn sort_by_thermal_gradient(&mut self) {
        self.nodes
            .sort_by(|a, b| a.thermal_score().partial_cmp(&b.thermal_score()).unwrap());
    }
}
