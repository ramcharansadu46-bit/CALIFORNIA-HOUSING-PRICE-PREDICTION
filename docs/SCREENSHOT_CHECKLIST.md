# Screenshot checklist (genuine screenshots only)

Nothing below has been created for you: screenshots must show **your own** machine, accounts and activity. Do not edit, mock up or reuse other people's screenshots. Save images as PNG in `docs/screenshots/` using the suggested file names.

## A. Genuine project screenshots (your computer)
- [ ] `A01_project_folder.png` - project folder tree in File Explorer / VS Code showing the structure.
- [ ] `A02_environment_setup.png` - terminal showing `pip install -r requirements.txt` finishing without errors.
- [ ] `A03_notebook_start.png` - the notebook opened in Jupyter/VS Code (title and objectives table visible).
- [ ] `A04_data_inspection.png` - output of the data-quality audit cells (missing values, capped counts).
- [ ] `A05_eda_figures.png` - one or two EDA figures rendered in the notebook (e.g. correlation heat-map).
- [ ] `A06_objective_results.png` - one result table per objective (O1-O6), at least one screenshot each.
- [ ] `A07_model_tuning.png` - the tuning/CV output cell (timings visible).
- [ ] `A08_model_comparison.png` - the model-comparison table and Figure 12.
- [ ] `A09_shap_pdp.png` - SHAP summary and partial-dependence outputs.
- [ ] `A10_saved_model_test.png` - the cell printing "saved model reproduces test predictions: OK".
- [ ] `A11_predict_script.png` - terminal output of `python -m src.predict data/processed/test.csv`.
- [ ] `A12_report_docx.png` - the DOCX open in Word showing cover page, table of contents (after updating fields) and one analysis page.

## B. GitHub repository screenshots (your GitHub account)
- [ ] `B01_repo_home.png` - repository front page with README rendered and the file list.
- [ ] `B02_repo_structure.png` - folders `data/`, `notebooks/`, `src/`, `figures/`, `results/`, `models/`, `report/`.
- [ ] `B03_notebook_rendered.png` - the notebook rendered on GitHub (figures visible).
- [ ] `B04_commit_history.png` - Commits page showing your real commit(s) and date.
- [ ] `B05_readme_results.png` - README results table.
- [ ] `B06_repo_url.png` - browser address bar showing your repository URL (put the URL in the report/README).

## C. LinkedIn screenshots (your LinkedIn account)
Only if you actually publish them:
- [ ] `C01_project_section.png` - the project added under Profile -> Projects (title, description, GitHub link).
- [ ] `C02_post.png` - your own post about the project (short summary, 1-2 figures, repository link).
- [ ] `C03_post_url.png` - the post URL and date visible.
Suggested post angle: the problem, the dataset, the three main findings (best model and RMSE, spatial effect, censoring effect), and a link to the repository. Do not claim results that are not in `results/metrics.json`.

## D. Where to use them
The LPU format has no appendix for screenshots; attach them after the references as an "Annexure" only if your supervisor asks, or submit them as separate evidence.
