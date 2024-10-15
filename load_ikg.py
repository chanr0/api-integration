# %%
import join_api
from join_api.common import *
import join_api.model
import json
from typing import List, Any
import pandas as pd

def _load_json(filename: str) -> Any:
    l.log(25, f"loading {filename}")
    with open(filename) as fd:
        return json.load(fd)

# l.info("parsing data")
source: str = "data/join_api/join_api_fs.json"
product_database_path: str = "data/join_api/deployment_targets.csv"
application_filter: List[str] = None


ikg = join_api.model.JoinApiKG_Base(product_database_path)
source_config = _load_json(source)
join_api.parser.parse(
    source_config, ikg, application_filter=application_filter, max_workers=1
)
ikg.postprocess()
ikg.postcheck(flags=join_api.F_PATCH)

# %%
commands, descriptions = [], []
app_dn, object_dn, operation_dn = [], [], []  # display_names
for _app in ikg.applications.values():
    for _obj in _app.objects.values():
        for _op in _obj.operations.values():
            app_dn.append(_app.display_name)
            object_dn.append(_obj.display_name)
            operation_dn.append(_op.display_name)
            commands.append(f"{_app.name}.{_obj.name}.{_op.name}")
            descriptions.append(_op.description)

df_descriptions_orig = pd.DataFrame(
    {
        "command": commands,
        "description": descriptions,
        "app_display_name": app_dn,
        "object_display_name": object_dn,
        "operation_display_name": operation_dn,
    }
    )
