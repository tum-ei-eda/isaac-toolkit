# Checking ELF basic blocks against Memgraph CDFGs

`isaac_toolkit.backend.memgraph.annotate_bb_weights` validates counts before
copying profile weights onto the CDFG. The demo's CDFG script already invokes
this command after graph generation, so the check runs automatically.

The validator joins the ELF-derived `llvm_bbs_new` table to `BB` records in the
requested Memgraph session using function name and BB name. It counts `INSTR`
nodes directly, including nodes without DFG edges. Synthetic `input`, `constant`,
and `label` operand nodes are excluded. The node lookup also matches the graph's
module and compiler stage. This is a consistency check for a CDFG stage close
to assembly; equal counts do not prove that instruction semantics match.

The default permitted difference in either direction is:

```
abs(graph_nodes - elf_instructions) <= 5 + 0.10 * elf_instructions
```

For example, a 100-instruction ELF block permits 85 through 115 graph nodes.
An empty graph for a nonempty ELF block always fails. Duplicate ELF identities
or multiple graph blocks sharing a function/BB identity also fail, rather than
combining modules or compiler stages into a misleading count.

Every run records a `bb_cdfg_validation` table artifact containing both counts,
the difference and allowance, status, selected-block flag, graph modules and
stages. Validation covers cold blocks as well as executed blocks. Missing
selected blocks on either side fail. ELF blocks outside the exported compilation
units are recorded as `not_exported`; other graph blocks missing from the ELF
are recorded as `missing_elf` and produce a coverage warning, since earlier
compiler stages can contain blocks eliminated before linking.

By default, a large mismatch stops the stage **before any weight writes**.
The CLI saves the diagnostic artifact in the session even on this failure.
Inspect the ELF, compiler stage, and database session label before proceeding.
The usual artifact replacement rules apply: rerun with `--force` to replace a
previous validation report.

To change the tolerance or explicitly continue after logging mismatches:

```sh
python -m isaac_toolkit.backend.memgraph.annotate_bb_weights \
    --session /path/to/session --label my-cdfg-session --force \
    --absolute-tolerance 3 --relative-tolerance 0.20 --mismatch-action warn
```

The Python `annotate_bb_weights` function accepts the corresponding
`absolute_tolerance`, `relative_tolerance`, and `mismatch_action` arguments.
Warning mode annotates comparable blocks, including count mismatches, but skips
missing or ambiguous identities. A session with no comparable blocks is never
silently accepted: error mode stops and warning mode logs it.
