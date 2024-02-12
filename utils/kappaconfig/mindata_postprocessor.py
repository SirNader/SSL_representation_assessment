import kappaconfig as kc

from .testrun_constants import TEST_RUN_EFFECTIVE_BATCH_SIZE, TEST_RUN_UPDATES_PER_EPOCH


class MinDataPostProcessor(kc.Processor):
    """
    hyperparams for specific properties in the dictionary and replace it such that the training duration is
    limited to a minimal configuration
    """

    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            # datasets
            if parent_accessor == "datasets":
                for key in node.keys():
                    wrapper = dict(
                        kind="subset_wrapper",
                        end_index=TEST_RUN_EFFECTIVE_BATCH_SIZE * TEST_RUN_UPDATES_PER_EPOCH,
                    )
                    if "dataset_wrappers" in node[key]:
                        node[key]["dataset_wrappers"].append(wrapper)
                    else:
                        assert isinstance(node[key], dict), (
                            "found non-dict value inside 'datasets' node -> probably wrong template "
                            "parameter (e.g. template.version instead of template.vars.version)"
                        )
                        node[key]["dataset_wrappers"] = [wrapper]
            elif parent_accessor == "effective_batch_size":
                parent[parent_accessor] = min(parent[parent_accessor], TEST_RUN_EFFECTIVE_BATCH_SIZE)
            elif parent_accessor == "relative_tjf_file_uris":
                parent[parent_accessor] = parent[parent_accessor][:5]
            # AudioSet min_segments & smaller size
            if isinstance(node, dict) and node.get("kind", None) == "audioset" and "split" in node:
                node["split"] = "min"
                transform = dict(kind="resize", size=[256, 128])
                if "x_transform" in node:
                    node["x_transform"].append(transform)
                else:
                    node["x_transform"] = [transform]
