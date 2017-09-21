Forked from @despoisj and adapted for genre classification (adapting readme on the fly)

# Deep Audio Classification

Required install:
```
eyed3
sox --with-lame
tensorflow
tflearn
```

- Create folder Data/Raw/
- Place your labeled .mp3 files in Data/Raw/

To create the song slices (might be long):

```
python main.py slice
```

To train the classifier (long too):

```
python main.py train
```

To test the classifier (fast):

```
python main.py test
```

- Most editable parameters are in the config.py file, the model can be changed in the model.py file.
- I haven't implemented the pipeline to label new songs with the model, but that can be easily done with the provided functions, and eyed3 for the mp3 manipulation. Here's the full pipeline you would need to use.

![alt tag](https://github.com/despoisj/DeepAudioClassification/blob/master/img/pipeline.png)
