// import("../pkg/sonors.js").catch(console.error);
import("../pkg/sonors").then(module => {
  let spectrogram = new module.Spectrogram(1024);
  const handleSuccess = function(stream) {
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(1024, 1, 1);

    source.connect(processor);
    processor.connect(context.destination);

    processor.onaudioprocess = function(e) {
      spectrogram.process_signal(e.inputBuffer.getChannelData(0));
    };
  }

  navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(handleSuccess);
});
// export class Spectrogram {
//     static __construct(ptr) {
//         return new Spectrogram(ptr);
//     }

//     constructor(ptr) {
//         this.ptr = ptr;
//     }

//     free() {
//         const ptr = this.ptr;
//         this.ptr = 0;
//         wasm.__wbg_spectrogram_free(ptr);
//     }

//     static new(arg0) {
//         const ret = wasm.spectrogram_new(arg0);
//         return Spectrogram.__construct(ret)
//     }

//     process_signal(arg0) {
//         wasm.spectrogram_process_signal(this.ptr, arg0);
//     }
// }

