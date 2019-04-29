library(FELLA)
set.seed(1)
graph <- buildGraphFromKEGGREST(
  organism = 'hsa', 
  filter.path = c('01100', '01200', '01210', '01212', '01230')
)
tmpdir <- paste0(tempdir(), 'my_database')
unlink(tmpdir, recursive=TRUE)
buildDataFromGraph(
  keggdata.graph = graph,
  databaseDir = tmpdir,
  internalDir = FALSE,
  matrices = 'diffusion',
  normality = 'diffusion',
  niter = 50
)

fella.data <- loadKEGGdata(
  databaseDir = tmpdir,
  internalDir = FALSE,
  loadMatrix = 'diffusion'
)

cat(getInfo(fella.data))