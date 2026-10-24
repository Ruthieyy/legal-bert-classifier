# Legal Contract Risk Classifier

Learning MVP: a notebook that fine-tunes `bert-base-uncased` to label short contract clauses as higher-risk (`1`) or lower-risk (`0`). The saved run prints **100% accuracy on 10 clauses**, then **100% on a 2-clause slice of those same clauses**. Read those numbers with the evaluation section below. They show that the training loop ran on this toy table. They are a weak estimate of how the model would do on new contracts.

## What the notebook does

The walkthrough is [`LEGAL BERT MVP.ipynb`](LEGAL%20BERT%20MVP.ipynb). From the top:

1. **Tokenizer check.** Load the `bert-base-uncased` tokenizer and print WordPiece ids for one clause ("The party shall not be liable for any damages whatsoever.").
2. **Public-data preview.** Load [CUAD](https://huggingface.co/datasets/theatticusproject/cuad) (`theatticusproject/cuad`). The saved run shows a train split of 511 PDFs and prints extracted text from the first PDF (the CUAD datasheet) and from the PDF at index 5 (affiliate-program terms, first two pages). Then load [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) SCOTUS (`coastalcph/lex_glue`, config `scotus`): 5,000 train / 1,400 validation / 1,400 test opinion-topic rows. The fine-tune in the next step uses the 10-row table only.
3. **Training table.** Ten clauses written in the notebook, each with a hand-assigned integer label. This project reads `1` as higher risk and `0` as lower risk (broad liability waivers, unilateral amendment, mandatory arbitration, and indemnity versus routine payment, governing-law, termination, and license language). The notebook stores the integers and does not include a written rubric. Six rows are `1` and four are `0`.
4. **Fine-tune.** `ContractDataset` tokenizes the ten clauses (`truncation`, `padding`, `max_length=128`). `BertForSequenceClassification` (`bert-base-uncased`, `num_labels=2`; the classification head starts randomly initialized) trains with Hugging Face `Trainer` for 10 epochs, batch size 4, `logging_steps=5`, `save_strategy="no"`.
5. **Score the training rows.** The same ten clauses go back through the model. Saved output: `Accuracy: 100.00%`.
6. **Second run, 8/2 split.** `train_test_split(..., test_size=0.2, random_state=42)` makes 8 training rows and 2 test rows. `Trainer` keeps the weights from step 5 and trains another 10 epochs on the 8. The 2 test rows are then scored. Saved output: `Accuracy: 100.00%`. Both are label `0`:
   - "Payment is due within 30 days of invoice."
   - "Either party may terminate this agreement with 30 days notice."

[`LEGAL_BERT_MVP.ipynb`](LEGAL_BERT_MVP.ipynb) is an earlier scratch notebook. Its saved output runs `!pip install transformers torch datasets` and the next cell raises `ModuleNotFoundError: No module named 'transformers'`. The fine-tune is in `LEGAL BERT MVP.ipynb`.

## Evaluation

| Run | Rows scored | Accuracy printed |
| --- | --- | --- |
| After 10 epochs on all 10 clauses | Those same 10 clauses | 100% |
| After 10 more epochs on the 8-row split | The other 2 clauses | 100% |

How to read that:

- The first number is training-set accuracy. Ten epochs on ten short sentences is enough for the model to memorize the table.
- The second number is two rows, both lower-risk. Predicting `0` for every clause also scores 100% on that pair. Predicting the majority label (`1`) for every clause scores 6/10 on the full table and 0/2 on the pair.
- Those two rows were already in the first training pass. The split is created afterward, and the second `Trainer` continues from the same weights, so the pair is not an unseen test set.
- The notebook reports accuracy only: no precision, recall, F1, confusion matrix, or baseline.

## How to run

The saved execution used Python 3.14, Transformers 5.9.0, Datasets 4.8.5, and PyTorch 2.9.1 on macOS. CPU is enough for this table. A current Python 3 venv with the packages below is the setup the notebook imports.

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers torch datasets pandas scikit-learn jupyter pdfplumber
jupyter notebook "LEGAL BERT MVP.ipynb"
```

Run the cells from the top. The CUAD and LexGLUE cells download from the Hugging Face Hub; CUAD includes contract PDFs and is the slow part. The Hub accepts anonymous downloads; a token only raises the rate limit. The toy fine-tune is 10 clauses for 10 epochs. Training creates a local `results/` directory and, with `save_strategy="no"`, does not write a reusable model checkpoint.

If an import fails immediately after the install cell, restart the kernel and run from the top again.

## Limitations / Next steps

**Limits of this run**

- Labeled data is 10 author-written sentences with author-assigned labels. CUAD and LexGLUE are previews; their labels never enter the loss.
- There is no annotation guide, second annotator, or link from a row back to a source contract.
- The 100% figures are the training table and a 2-row, single-class slice of it, scored after those rows were already used to update the weights.
- The encoder is general-domain `bert-base-uncased` with a 128-token window, so longer clauses are truncated. No legal-domain checkpoint is loaded.
- Weights stay in the notebook session. There is no saved model and no inference script.

**Next steps**

- Write down the risk definition and apply it to clauses from a public corpus (CUAD's category spans are one source), with a second pass on a sample of the labels.
- Hold out a stratified test set before any training. Start each reported run from a freshly initialized classification head so test clauses stay unseen.
- Report accuracy beside a majority-class baseline, plus per-class precision, recall, and F1. Treat a percentage on two rows as an anecdote until the held-out set is large enough to move the estimate.
- Save the tokenizer and model, and add a short script that scores a clause from the command line.
- After the labeled set can support it, try a legal-domain encoder and a longer max length.
