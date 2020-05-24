import tensorflow as tf

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

model = tf.keras.models.load_model('./model.h5')

print(model.outputs)
print('#####################')
print(model.inputs)

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
export_path = 'gs://dlaas-model/model/export/1/'
tf.saved_model.save(model, export_dir=export_path)
#        inputs={'input_image': model.input},
#        outputs={t.name: t for t in model.outputs})
