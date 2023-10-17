// worker.js
self.onmessage = function(event) {
    // This function is executed when the worker receives a message from the main thread.
    const dataFromMain = event.data;
  
    // Perform some processing or calculations
    const result = processData(dataFromMain);
  
    // Send the result back to the main thread
    self.postMessage(result);
  };
  
  function processData(data) {
    // Perform some data processing or calculations here
    // ...
  
    return result;
  }
  