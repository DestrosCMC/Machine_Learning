from keras.optimizers import SGD

def get_new_model(input_shape = input_shape):
  model = Sequential()
  model.add(Dense(100, activation='relu', input_shape = input_shape))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  return(model)

lr_to_test = [.000001, 0.01, 1]

# loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)
