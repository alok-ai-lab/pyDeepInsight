# Datasets

<a id='tcga.rnaseq_fpkm_uq.example'></a>
# tcga.rnaseq_fpkm_uq.example.txt.gz

A subset of the TCGA [RNAseq FPKM UQ](https://docs.gdc.cancer.gov/Encyclopedia/pages/HTSeq-FPKM-UQ/) data from the 
original manuscipt. The example data was generated from a random selection of 200 samples from each of the cancer types
BRCA, KIRC, and COAD (total 600 samples). Then the 5,000 genes with the largest standard variation were selected.

<a id='curatedTCGAData_RNASeq2GeneNorm-20160128'></a>
# curatedTCGAData_RNASeq2GeneNorm-20160128.txt.gz

The following R code generates a file containing primary tumor RNASeq gene-level expression data for the all TCGA disease/cancer types with at least 500 unique patients as annotated by the Bioconductor package [curatedTCGAData](https://bioconductor.org/packages/release/data/experiment/html/curatedTCGAData.html). The final dataset contains 4,550 samples and 19,383 genes across 8 TCGA cancer types and is just under 250MB in size.


```R
suppressMessages({
    library('tidyverse')
    library('MultiAssayExperiment')
    library('curatedTCGAData')
    library('TCGAutils')
    library('caret')
})
```

Get all disease codes available for TCGA


```R
data(diseaseCodes)
dcodes <- diseaseCodes %>%
dplyr::filter(Available == 'Yes') %>%
pull(Study.Abbreviation) %>%
glimpse()
```

     chr [1:33] "ACC" "BLCA" "BRCA" "CESC" "CHOL" "COAD" "DLBC" "ESCA" "GBM" ...


Download all RNASeq2GeneNorm data for all disease codes


```R
assays = c('RNASeq2GeneNorm')

suppressMessages(
    mae_all <- purrr::map(dcodes, ~curatedTCGAData(
        .x,
        assays = assays,
        version = '2.0.1',
        dry.run=FALSE,
        verbose=FALSE
    ) %>% TCGAprimaryTumors())
)
names(mae_all) <- dcodes
```

Get sample mappings for all RNASeq2GeneNorm data


```R
samples <- purrr::map(dcodes, ~as_tibble(sampleMap(mae_all[[.x]])) %>% mutate(cancer=.x)) %>%
bind_rows() %>%
glimpse()
```

    Rows: 9,889
    Columns: 4
    $ assay   <fct> ACC_RNASeq2GeneNorm-20160128, ACC_RNASeq2GeneNorm-20160128, AC…
    $ primary <chr> "TCGA-OR-A5J1", "TCGA-OR-A5J2", "TCGA-OR-A5J3", "TCGA-OR-A5J5"…
    $ colname <chr> "TCGA-OR-A5J1-01A-11R-A29S-07", "TCGA-OR-A5J2-01A-11R-A29S-07"…
    $ cancer  <chr> "ACC", "ACC", "ACC", "ACC", "ACC", "ACC", "ACC", "ACC", "ACC",…


Identify disease codes with greater than 500 unique patient IDs


```R
sample_select <- samples %>%
distinct(primary, cancer) %>%
group_by(cancer) %>%
summarize(n=n()) %>%
filter(n >= 500) %>%
glimpse() %>%
pull(cancer)
```

    Rows: 8
    Columns: 2
    $ cancer <chr> "BRCA", "HNSC", "KIRC", "LGG", "LUAD", "LUSC", "THCA", "UCEC"
    $ n      <int> 1093, 520, 533, 516, 515, 501, 501, 545


Subset expression and sample map data


```R
mae_all <- mae_all[sample_select]

sample_map <- purrr::map(sample_select, ~as_tibble(sampleMap(mae_all[[.x]])) %>% mutate(cancer=.x)) %>%
bind_rows() %>%
dplyr::select(patient_id=primary, colname, cancer) %>%
glimpse()
```

    Rows: 4,725
    Columns: 3
    $ patient_id <chr> "TCGA-3C-AAAU", "TCGA-3C-AALI", "TCGA-3C-AALJ", "TCGA-3C-AA…
    $ colname    <chr> "TCGA-3C-AAAU-01A-11R-A41B-07", "TCGA-3C-AALI-01A-11R-A41B-…
    $ cancer     <chr> "BRCA", "BRCA", "BRCA", "BRCA", "BRCA", "BRCA", "BRCA", "BR…


Generate final expression dataframe with patient ID and cancer label. Removes genes with near zero variance.


```R
exp_df <- purrr::map(sample_select, ~mae_all[[.x]][[1]] %>% assay() %>% t() %>% as.data.frame()) %>%
bind_rows()

nzv_cols <- nearZeroVar(exp_df)
if(length(nzv_cols) > 0) exp_df <- exp_df[, -nzv_cols]

exp_df <- exp_df %>%
rownames_to_column("colname") %>%
inner_join(sample_map, by='colname') %>%
dplyr::select(-colname) %>%
dplyr::select(patient_id, cancer, everything())
```


```R
options(repr.matrix.max.rows=10, repr.matrix.max.cols=15)
exp_df
```


<table class="dataframe">
<caption>A data.frame: 4550 × 19385</caption>
<thead>
        <tr><th scope=col>patient_id</th><th scope=col>cancer</th><th scope=col>A1BG</th><th scope=col>A1CF</th><th scope=col>A2BP1</th><th scope=col>A2LD1</th><th scope=col>A2ML1</th><th scope=col>A2M</th><th scope=col>⋯</th><th scope=col>ZYG11A</th><th scope=col>ZYG11B</th><th scope=col>ZYX</th><th scope=col>ZZEF1</th><th scope=col>ZZZ3</th><th scope=col>psiTPTE22</th><th scope=col>tAKR</th></tr>
        <tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
        <tr><td>TCGA-3C-AAAU</td><td>BRCA</td><td>197.0897</td><td>0.0000</td><td>0.0000</td><td>102.9634</td><td>1.3786</td><td> 5798.375</td><td>⋯</td><td>258.4941</td><td>1208.3738</td><td>3507.248</td><td>1894.9342</td><td>1180.4565</td><td>  1.7233</td><td>0</td></tr>
        <tr><td>TCGA-3C-AALI</td><td>BRCA</td><td>237.3844</td><td>0.0000</td><td>0.0000</td><td> 70.8646</td><td>4.3502</td><td> 7571.979</td><td>⋯</td><td>198.4774</td><td> 603.5889</td><td>5504.622</td><td>1318.6514</td><td> 406.7428</td><td>926.5905</td><td>0</td></tr>
        <tr><td>TCGA-3C-AALJ</td><td>BRCA</td><td>423.2366</td><td>0.9066</td><td>0.0000</td><td>161.2602</td><td>0.0000</td><td> 8840.399</td><td>⋯</td><td>331.8223</td><td> 532.1850</td><td>5458.749</td><td> 942.8830</td><td> 509.5195</td><td> 35.3581</td><td>0</td></tr>
        <tr><td>TCGA-3C-AALK</td><td>BRCA</td><td>191.0178</td><td>0.0000</td><td>0.0000</td><td> 62.5072</td><td>1.6549</td><td>10960.219</td><td>⋯</td><td>175.4241</td><td> 607.3645</td><td>5691.353</td><td> 781.1336</td><td> 700.8688</td><td> 66.6115</td><td>0</td></tr>
        <tr><td>TCGA-4H-AAAK</td><td>BRCA</td><td>268.8809</td><td>0.4255</td><td>3.8298</td><td>154.3702</td><td>3.4043</td><td> 9585.443</td><td>⋯</td><td> 14.0426</td><td> 775.7447</td><td>4041.702</td><td> 831.9149</td><td> 881.7021</td><td>187.2340</td><td>0</td></tr>
        <tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋱</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
        <tr><td>TCGA-FI-A2EW</td><td>UCEC</td><td>247.5762</td><td>0</td><td>11.2360</td><td> 90.1605</td><td> 22.4719</td><td> 5007.255</td><td>⋯</td><td>231.1396</td><td> 601.9262</td><td>4181.380</td><td>1120.3852</td><td>929.3740</td><td>117.1750</td><td>0</td></tr>
        <tr><td>TCGA-FI-A2EX</td><td>UCEC</td><td>141.3892</td><td>0</td><td> 0.9833</td><td>118.6454</td><td> 20.6494</td><td> 3527.061</td><td>⋯</td><td>158.3119</td><td> 620.4645</td><td>6604.850</td><td> 984.2868</td><td>524.1008</td><td>705.0286</td><td>0</td></tr>
        <tr><td>TCGA-FI-A2F4</td><td>UCEC</td><td> 38.4000</td><td>0</td><td> 0.0000</td><td>119.6356</td><td>644.4444</td><td> 4674.711</td><td>⋯</td><td>  2.6667</td><td> 584.0000</td><td>5286.222</td><td>1259.5556</td><td>619.5556</td><td> 32.8889</td><td>0</td></tr>
        <tr><td>TCGA-FI-A2F8</td><td>UCEC</td><td>147.5587</td><td>0</td><td> 4.4206</td><td> 87.8589</td><td> 44.2057</td><td>11065.004</td><td>⋯</td><td> 86.2012</td><td> 669.7169</td><td>9367.195</td><td>1043.2553</td><td>689.6094</td><td> 95.0423</td><td>0</td></tr>
        <tr><td>TCGA-FI-A2F9</td><td>UCEC</td><td>192.8314</td><td>0</td><td> 1.2870</td><td>134.4273</td><td>  3.8610</td><td> 8995.882</td><td>⋯</td><td>  9.0090</td><td>1068.2111</td><td>5338.481</td><td>1002.5740</td><td>711.7117</td><td> 37.3230</td><td>0</td></tr>
</tbody>
</table>




```R
write_delim(exp_df, "./curatedTCGAData_RNASeq2GeneNorm-20160128.txt.gz", delim="\t")
```