# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

<img width="1018" height="336" alt="Screenshot 2026-05-19 091532" src="https://github.com/user-attachments/assets/0b0bae35-13ea-4414-8acf-3fcf4def1213" />


## DESIGN STEPS
### STEP 1: 
Load and normalize data, create sequences.

### STEP 2:
Convert data to tensors and set up DataLoader.

### STEP 3:
Define the RNN model architecture.

### STEP 4:
Summarize, compile with loss and optimizer.

### STEP 5:
Train the model with loss tracking.

### STEP 6:
Predict on test data, plot actual vs. predicted prices. 


## PROGRAM

### Name: Darshan V

### Register Number: 212224230050


```python
# Define RNN Model
class RNNModel(nn.Module):
    # write your code here
    def __init__(self,input_size=1, hidden_size=64, num_layers=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc=nn.Linear(hidden_size, output_size)
    def forward(self,x):
      out,_=self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out

# Train the Model
# Write your code here
def train_model(model,train_loader, criterion, optimizer,epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch, y_batch in train_loader:
      x_batch, y_batch= x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/ len(train_loader))
    print(f"Epoch [{ epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    # Plot training loss
  print('Name: Amruthavarshini Gopal')
  print('Register Number: 212223230013')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model(model, train_loader, criterion, optimizer)

```

## OUTPUT

### Training Loss Over Epochs Plot

<img width="550" height="549" alt="image" src="https://github.com/user-attachments/assets/89a09257-e5a2-4f4f-be98-6bb670bb17de" />



### True Stock Price, Predicted Stock Price vs time

<img width="633" height="371" alt="image" src="https://github.com/user-attachments/assets/3b889c7b-4325-45ac-a25f-3769a72148a9" />



### Predictions
<img width="232" height="51" alt="image" src="https://github.com/user-attachments/assets/97b54028-7e76-4594-a17c-cfb037eebd45" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
