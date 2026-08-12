// Read-only query used for the thesis demo screenshot.
// The selected cluster is the real cross-article case cat2_07.
MATCH p=(source:Dieu {id: 'lu_t_giao_d_ch_i_n_t_2023_D37'})-[:THAM_CHIEU]->(target:Dieu)
WHERE target.id IN [
  'lu_t_giao_d_ch_i_n_t_2023_D15',
  'lu_t_giao_d_ch_i_n_t_2023_D16',
  'lu_t_giao_d_ch_i_n_t_2023_D17',
  'lu_t_giao_d_ch_i_n_t_2023_D18'
]
RETURN p
LIMIT 10;
