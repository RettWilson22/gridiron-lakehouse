-- In every scoring format, floor <= projection <= ceiling.
select projection_key
from {{ ref('stg_projections') }}
where floor_ppr > proj_ppr + 0.000001
   or proj_ppr > ceiling_ppr + 0.000001
   or floor_half > proj_half + 0.000001
   or proj_half > ceiling_half + 0.000001
   or floor_std > proj_std + 0.000001
   or proj_std > ceiling_std + 0.000001
