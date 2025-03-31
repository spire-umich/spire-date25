# SPIRE - DATE 2025 Repository

SPIRE is a performance model that can help identify μarch-level performance bottlenecks. It was originally presented in the DATE 2025 paper "SPIRE: Inferring Hardware Bottlenecks from Performance Counter Data". The paper's pre-recorded presentation is available [here](https://youtu.be/MoClGXZnwTE).

Note: This repo is in a preliminary state and will be finalized with additional documentation after DATE 2025 concludes.

This repository contains a Python implementation of the SPIRE model ([spire.py](spire.py)) and a simple program demonstrating its usage ([demo.py](demo.py)).

# Dependencies

Version: (Oldest tested - Newest tested)
- Python3 (3.8.10 - 3.13.2)
- [NumPy](https://numpy.org/) (1.22.4 - 2.2.3)
- [Networkx](https://networkx.org/) (2.8.8 - 3.4.2)

# Demo Usage

## Quick Start

Download and unzip the samples used in the SPIRE paper (available [here](https://drive.google.com/file/d/15mx2yblwg8_gqgl9Y4i93JB8Up79R4HE/view?usp=sharing) - 12MB download, 28MB uncompressed)

Train the SPIRE model:
```
python3 demo.py train model.pickle demo_data/train.csv
```

<details>
<summary>Expected Output</summary>

```
Importing "demo_data\train.csv"...
  Imported 1,265,216 samples for 424 metrics

Training data:
  Time is "cpu_clk_unhalted.thread"
  Work is "inst_retired.any"
  424 Metrics
  1,265,216 Samples

Training SPIRE model...
Saving model to "model.pickle"...
```

</details>

Analyze the 4 test workloads:
```
python3 demo.py analyze model.pickle demo_data/tnn.csv demo_data/scikit.csv demo_data/onnx.csv demo_data/parboil.csv
```
<details>
<summary>Expected Output</summary>

```
Loading model from "model.pickle"...
  Time is "cpu_clk_unhalted.thread"
  Work is "inst_retired.any"
  424 Metrics

Importing "demo_data/tnn.csv"...
  Imported 5,936 samples for 424 metrics
Importing "demo_data/scikit.csv"...
  Imported 27,560 samples for 424 metrics
Importing "demo_data/onnx.csv"...
  Imported 12,720 samples for 424 metrics
Importing "demo_data/parboil.csv"...
  Imported 8,056 samples for 424 metrics

10 Lowest-Throughput Metric(s) for "demo_data/tnn.csv":
  0.77  1.00  frontend_retired.latency_ge_2_bubbles_ge_3
  1.03  0.00  idq.dsb_cycles
  1.62  1.00  idq_uops_not_delivered.cycles_le_2_uop_deliv.core
  1.64  1.00  frontend_retired.latency_ge_2_bubbles_ge_1
  1.66  1.00  frontend_retired.dsb_miss
  1.70  1.00  idq_uops_not_delivered.cycles_le_1_uop_deliv.core
  1.71  0.00  idq.dsb_uops
  1.74  1.00  idq_uops_not_delivered.core
  1.92  1.00  idq_uops_not_delivered.cycles_le_3_uop_deliv.core
  1.94  1.00  frontend_retired.latency_ge_2_bubbles_ge_2


10 Lowest-Throughput Metric(s) for "demo_data/scikit.csv":
  1.82  1.00  uops_retired.stall_cycles
  1.90  1.00  exe_activity.exe_bound_0_ports
  1.93  1.00  idq_uops_not_delivered.cycles_fe_was_ok
  2.05  1.00  cycle_activity.cycles_mem_any
  2.07  1.00  exe_activity.1_ports_util
  2.09  1.00  int_misc.recovery_cycles
  2.10  1.00  br_misp_retired.all_branches
  2.10  1.00  idq.dsb_cycles
  2.10  1.00  idq.all_dsb_cycles_any_uops
  2.10  1.00  int_misc.recovery_cycles_any


10 Lowest-Throughput Metric(s) for "demo_data/onnx.csv":
  0.04  1.00  uops_issued.vector_width_mismatch
  0.23  1.00  longest_lat_cache.miss
  0.23  1.00  cycle_activity.cycles_l1d_miss
  0.24  1.00  cycle_activity.cycles_mem_any
  0.24  1.00  idq_uops_not_delivered.cycles_fe_was_ok
  0.24  1.00  resource_stalls.any
  0.24  1.00  uops_executed.stall_cycles
  0.24  1.00  l1d_pend_miss.pending_cycles
  0.24  1.00  cycle_activity.stalls_total
  0.25  1.00  cycle_activity.stalls_l1d_miss


10 Lowest-Throughput Metric(s) for "demo_data/parboil.csv":
  0.84  1.00  exe_activity.1_ports_util
  1.00  1.00  mem_inst_retired.lock_loads
  1.33  1.00  idq.ms_switches
  1.37  1.00  cycle_activity.cycles_mem_any
  1.37  1.00  idq_uops_not_delivered.cycles_fe_was_ok
  1.53  1.00  uops_issued.stall_cycles
  1.57  1.00  resource_stalls.any
  1.57  1.00  idq.ms_dsb_cycles
  1.62  1.00  uops_executed.core_cycles_ge_1
  1.62  1.00  uops_executed.cycles_ge_1_uop_exec
```

</details>

Interpreting the analysis outputs:
```
10 Lowest-Throughput Metric(s) for "demo_data/tnn.csv":
  0.77  1.00  frontend_retired.latency_ge_2_bubbles_ge_3
  1.03  0.00  idq.dsb_cycles
```
The first column gives SPIRE's estimated throughput (IPC in this case) for that row's metric. Column two is the fraction of time the metric's samples were *left* of the roofline's peak. The final column gives the metric's full name.

For this workload, the ```frontend_retired.latency_ge_2_bubbles_ge_3``` metric had the lowest estimated IPC (0.77) and ```idq.dsb_cycles``` had the second lowest (1.03).
All of the former's samples are *left* of its roofline's peak. Thus, the model is suggesting that this metric's operational intensities should be increased to improve performance (*i.e.*, the metric's values should be reduced).
In contrast, all samples for ```idq.dsb_cycles``` are *right* of the roofline's peak. So, reducing this metric's operational intensities (*i.e.*, increasing its values) might improve performance.

### Differences With the Paper's Results
A bugfix in this repo's SPIRE implementation results in slightly different IPC estimations for some metrics reported in our paper (up to a 0.003 IPC change). We note the visible differences below. Notably, they would not have changed our interpretation of SPIRE's outputs for the test workloads.

<details>
<summary>Visible Differences</summary>

**TNN**:

None

**Scikit**:
| Paper    |                                         |     | Diff     |                                         |
| :------- | :-------------------------------------- | --- | :------- | :-------------------------------------- |
| **IPC**  | **Metric**                              |     | **IPC**  | **Metric**                              |
| **1.81** | uops_retired.stall_cycles               | ->  | **1.82** |                                         |
| 1.90     | exe_activity.exe_bound_0_ports          |
| **1.92** | idq_uops_not_delivered.cycles_fe_was_ok | ->  | **1.93** |                                         |
| 2.05     | cycle_activity.cycles_mem_any           |
| 2.07     | exe_activity.1_ports_util               |
| 2.09     | int_misc.recovery_cycles                |
| **2.09** | br_misp_retired.all_branches            | ->  | **2.10** |                                         |
| 2.10     | idq.dsb_cycles                          |
| 2.10     | **int_misc.recovery_cycles_any**        | ->  |          | **idq.all_dsb_cycles_any_uops**         |
| 2.10     | **idq.all_dsb_cycles_any_uops**         | ->  |          | **int_misc.recovery_cycles_any**        |


**ONNX**:
| Paper    |                                         |     | Diff     |                                         |
| :------- | :-------------------------------------- | --- | :------- | :-------------------------------------- |
| **IPC**  | **Metric**                              |     | **IPC**  | **Metric**                              |
| 0.04     | uops_issued.vector_width_mismatch       |
| 0.23     | longest_lat_cache.miss                  |
| 0.23     | cycle_activity.cycles_l1d_miss          |
| 0.24     | cycle_activity.cycles_mem_any           |
| 0.24     | idq_uops_not_delivered.cycles_fe_was_ok |
| 0.24     | resource_stalls.any                     |
| 0.24     | uops_executed.stall_cycles              |
| 0.24     | cycle_activity.stalls_total             | ->  |          | **l1d_pend_miss.pending_cycles**        |
| **0.25** | cycle_activity.stalls_l1d_miss          | ->  | **0.24** | **cycle_activity.stalls_total**         |
| 0.25     | l1d_pend_miss.pending_cycles            | ->  |          | **cycle_activity.stalls_l1d_miss**      |

**Parboil**:

None

</details>
