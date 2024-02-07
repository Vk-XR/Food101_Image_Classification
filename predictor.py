def food101_predictor(image_path):

 import tensorflow as tf
 import matplotlib.pyplot as plt

 # Make a function to preprocess images
 def preprocess_image(image,label,img_shape = 224):
  """
  Converts image datatype from `uint8` to `float32` and reshapes image to [img_shape,img_shape,colour_channels]
  """

  image = tf.image.resize(image,[img_shape,img_shape]) # Reshape
  # image = image/255. # scale image values (required for models like ResNet)
  return tf.cast(image,tf.float32),label # return (float32_image,label) tuple

 CHECKPOINT_PATH = "model_checkpoints/cp.ckpt"
 class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
  'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
  'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']

 input_shape = (224,224,3)
 base_model = tf.keras.applications.EfficientNetV2B1(include_top = False,pooling = 'avg')
 base_model.trainable = False
 # Create functional model
 inputs = tf.keras.layers.Input(shape = input_shape,name = "input_layer")
 # x = tf.keras.layers.Rescaling(1./255)(x) # For models like ResNet
 x = base_model(inputs,training = False)
 x = tf.keras.layers.Dense(101)(x)
 outputs = tf.keras.layers.Activation("softmax",dtype = tf.float32,name = "softmax_float32")(x)
 model = tf.keras.Model(inputs,outputs)

 base_model.trainable = True

 for layer in base_model.layers[:-10]:
  layer.trainable = False

 model.load_weights(CHECKPOINT_PATH)

 my_img = tf.io.read_file(image_path)
 img_to_tensor = tf.io.decode_jpeg(my_img,channels = 3,fancy_upscaling = False)

 plt.figure(figsize = (6,8))
 predicted_label = model.predict(tf.expand_dims(preprocess_image(img_to_tensor,2)[0],axis = 0)).argmax(axis = 1).item()
 plt.imshow(img_to_tensor)
 plt.axis('off')
 plt.title(f"Predicted: {class_names[predicted_label]}")
 plt.show()


food101_predictor(input("Enter path to an food image: "))
