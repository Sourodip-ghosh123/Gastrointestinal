final_preds = (vgg_outcomes.drop('image_id', axis=1)*0.33 + densenet_outcomes.drop('image_id', axis=1)*0.33 + cnn_outcomes.drop('image_id', axis=1)*0.33).to_numpy()
final_preds = softmax(final_preds).argmax(1)

accuracy_score(final_preds, df['label'].values)
