2026-10-07: the updated intergen.ckpt is a genuine native InterGen generator.
weight_audit.json records actual key/shape comparison against official FreeMotion.
No cross-attention -> self-attention semantic mapping was applied.
Single is missing48 condition_sa tensors; multi also lacks learned Stage2
control/guidance parameters. No FreeMotion probe was generated.
evaluator_sanity.* uses real test motions only, not generated probe motions.
The user redirected experiments to ../intergen_vs_freemotion/.
