// import("../pkg/sonors.js").catch(console.error);
import("../pkg/sonors").then(module => {
  let spectrogram = new module.Spectrogram(1024, navigator.vendor == "Apple Computer, Inc.");
  const streamMedia = function(stream) {
    let AudioContext = window.AudioContext || window.webkitAudioContext;

    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(1024, 1, 1);

    source.connect(processor);
    processor.connect(context.destination);

    processor.onaudioprocess = function(e) {
      spectrogram.process_signal(e.inputBuffer.getChannelData(0));
    };
  }

  if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
  }

  // based off of MDN
  if (navigator.mediaDevices.getUserMedia === undefined) {
    navigator.mediaDevices.getUserMedia = function(constraints) {
      let getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

      if (!getUserMedia) {
        return Promise.reject(new Error("getUserMedia is not implemented in this browser."));
      }

      return new Promise(function(resolve, reject) {
        getUserMedia.call(navigator, constraints, resolve, reject);
      });
    }
  }


  navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(streamMedia);
});

