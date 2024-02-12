from utils.factory import instantiate


def collator_from_kwargs(kind, **kwargs):
    return instantiate(
        module_names=[f"kappadata.collators.{kind}"],
        type_names=[kind],
        **kwargs,
    )
