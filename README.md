Latest updates:


### Thursday Log (Dem)
@Benjie
- in v3, I can't make the graph look good. I duno, you try. Maybe do a mean calculation or smthn...
- in v4, it might be good to make a heatmap or a 3d graph? again, no damn clue how.... I have the data working there sort of though...
- implement performance threshold?? I'm not sure how...
- From Callum, if you have time: https://pyquestions.com/how-to-extract-data-from-matplotlib-plot. He can analyse the data in excel then...

function edits
- show_letter (square roots the length and trims the rest. Also displays values clearly)
- changed noise to continuous
- added potential tanh normalisation (VERY BROKEN, DONT TOUCH)

bug:
- we only get data from last image
- [fixed, I think] self.size was wrong

realisations:
- Calum's idea of over-training with noise improved attractors ALOT
- network size affects stuff....
    - tanh is "harsher" because higher inputs are possible due to bigger weight input!
    - stopping error is affected?



### Wednesday Log (Dem)
new functions:
- HopfieldNetwork.inference_step(X_input, iterations, "af_string")
    - use instead of IS_tanh_sync()
- sync_tanh(X, iterations): updating all at once
- async_tanh(X, iterations): update 1 at a time randomly
- async_n_tanh(X, iterations): update n at a time randomly (equivalent to sync_tan when n = len(X))

problems:
- [fixed] validate doesn't work as it should
- letter_res doesn't properly resize letters (although probably doesn't rly matter)

bug fixes:
- update functions had X.dot(weights) instead of weights.dot(X)
