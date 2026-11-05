# Data

This folder is where the ESS source file goes. **No ESS data are distributed with this repository.**

The analysis uses the **European Social Survey Round 11, integrated file, edition 4.2** (ESS ERIC, 2023/2024).

1. Register (free) and download the SPSS file from the ESS Data Portal: <https://ess.sikt.no/>.
2. Save it here as `data/ESS11e04_2.sav`.
3. Check that it is the same file used for the paper:

   ```
   sha256sum data/ESS11e04_2.sav
   # 35809b1883495d8e1cb3ed604f56c2ddf80c2666e9be11332a7d3502d3d631ae
   ```

The ESS data are subject to the ESS conditions of use: they may not be redistributed. Everything the pipeline derives at respondent level (`local/`, prediction files, fold assignments) is git-ignored and must stay on your own computer.
