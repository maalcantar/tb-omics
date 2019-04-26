library(tmod)

diff_metab = data.frame(data.table::fread('./tb-omics/data/tmod_diff.txt', header=TRUE, sep = "\t"), row.names=1)
corr_metab = data.frame(data.table::fread('./tb-omics/data/tmod_corr.txt', header=TRUE, sep = "\t"), row.names=1)
weights_metab = data.frame(data.table::fread('./tb-omics/data/Weights_HMDB.csv', header=TRUE))
weights_metab = weights_metab[order(-abs(weights_metab$X0)), ]


all_metabs = corr_metab$metab_ids
weights_list = weights_metab$metab_ids
weights_list = c(weights_list, diff_metab$metab_ids[!diff_metab$metab_ids %in% weights_list])

data(modmetabo)
tmod_diff <- tmodCERNOtest(diff_metab$metab_ids, mset=modmetabo)
tmod_corr <- tmodCERNOtest(corr_metab$metab_ids, mset=modmetabo)
tmod_weights <- tmodCERNOtest(weights_list, mset=modmetabo)

m2path_stats = data.frame(data.table::fread('./tb-omics/data/stats_table.txt', header=TRUE, sep = "\t"), row.names=1)
res <- m2path::m2path(score_set=m2path_stats,
                      output='./tb-omics/data', 
                      mapper_pathway2hmdb_file='./tb-omics/data/smpdb_metabolites.csv', 
                      mapper_oldhmdb_newhmdb_file= './tb-omics/data/ParsedHMDB_v4.0.csv')
