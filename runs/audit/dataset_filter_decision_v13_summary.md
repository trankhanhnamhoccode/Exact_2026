# Dataset filter decision after v13
## Decision counts
- dev: 48
- gold_wrong_existing: 6

## Dev priorities
- two-charge geometry package: 13
- charged distribution formulas: 7
- geometry/vector field package: 5
- symbolic support: 5
- collinear two-charge package: 4
- force-from-field final intent: 3
- geometry inverse package: 2
- collinear multi-charge package: 2
- capacitor state package: 1
- symbolic/geometry support: 1
- mechanics/electric kinematics: 1
- scalar inverse mechanics/electric: 1
- parser/scalar formula: 1
- extend v13 parser: 1
- triangle/vector field package: 1

## Decision
No new high-confidence gold-wrong rows were added beyond the existing 6 quality flags. The remaining 48 failures are development targets. The best next batch is the two-charge geometry package, because it covers perpendicular-bisector/midpoint/collinear vector-field cases with the same 2D/1D vector-superposition root cause.
