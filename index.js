import fs from "fs"
import csv from "csv-parser"
import xls from "xlsx"
import tf from "@tensorflow/tfjs"

// author : Ahmad Basofi Riswanto 
// NIM : 210403010014
// Tugas Akhir Matakuliah Kecerdasan Buatan

// initialize global variables
let dataPenyakit, dataGejala, dataDiagnosa;


// program utama
async function run() {
    
    // membaca data gejala, peyakit dan dataset diagnosa dari file csv yang ada di folder data 
    // dan menyimpan hasil read ke dalam variabel global
    const gejala = await read('data/gejala.csv');
    dataGejala = gejala;
    const penyakits = await read('data/penyakit.csv');
    dataPenyakit = penyakits;
    // untuk dataset yang digunakan saat ini adalah dataset yang sudah di kurangi hingga 42 row 
    // untuk mnggunakan dataset yang aslinya bisa mengganti ke datasetFull.csv
    const data = await read('data/dataset.csv');
    dataDiagnosa = data;

    // mapping dataset diagnosa dengan gejala dan penyakit menjadi bentuk yang bisa digunakan
    // untuk training model JST
    const datas = dataDiagnosa.map(dg => {
        return {
            gejala: getPenyakitIndecsByNama(dg.Disease),
            penyakit: dg.Disease
        }
    })

    // Memisahkan hasil mapping menjadi penyakit dan gejala yang disebabkan 
    const gejalaData = datas.map(entry => entry.gejala);
    const penyakitLabels = datas.map(entry => entry.penyakit);

    // Convert the dataset to tensors2d dan hot encoding agar data bisa digunakan untuk training model JST
    const gejalaTensor = tf.tensor2d(gejalaData);
    const penyakitTensor = tf.oneHot(tf.cast(tf.tensor1d(penyakitLabels.map(label => getLabelIndex(label))),'int32'), 3);

    // membuat model JST dengan 2 layer dengan lapiasn pertama dengan 16 neuron dan layer kedua dengan 3 neuron
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [gejalaTensor.shape[1]] }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compile the model dengan optimizer adam dan loss categoricalCrossentropy
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    
    // melatih model dengan data gejala dan penyakit yang sudah di mapping 
    const epochs = 50;
    const batchSize = 4;
    model.fit(gejalaTensor, penyakitTensor, { epochs, batchSize })
        .then(() => {
            // kode setelah model selesai dilatih , bisa digunakan untuk validasi dan prediksi 
            const gejalaExample  = [ "chills","vomiting","high_fever","sweating"];
            // Melakukan prediksi dengan gejala yang sudah diberikan sebelumnya 
            const gejalaTensorp = tf.tensor2d([convertGejalaToIndecs(...gejalaExample)]);
            const prediction = model.predict(gejalaTensorp);
            const predictedLabels = tf.argMax(prediction,  1).arraySync();
            const predictedPenyakit = predictedLabels.map(labelIndex => getLabelFromIndex(labelIndex));
            console.log(`prediksi penyakit dari gejala [${gejalaExample}] adalah ${predictedPenyakit}`)
        });
}

run()


async function read(path) {
    const data = xls.utils.sheet_to_json(xls.readFile(path).Sheets['Sheet1']);
    return data;
}


function getPenyakitIndecsByNama(namaPenyakit) {
    const diagnosa = dataDiagnosa.find(dd => p(dd.Disease) === p(namaPenyakit))
    if (!diagnosa) {
        console.log("tolol kali ah")
    }
    const gejalaPenyakit = Object.entries(diagnosa).filter((([k, v]) => k != 'Disease')).map(([k, v]) => p(v))
    return dataGejala.map(gejala => gejalaPenyakit.includes(p(gejala.namagejala)) ? 1 : 0)
}

function convertGejalaToIndecs(...gejala) {
    return dataGejala.map(gej => p(gej.namagejala)).map(gej => gejala.includes(p(gej)) ? 1 : 0)
}

function getLabelIndex(label) {
    const penyakitLabels = dataPenyakit.map(p => p.Disease);
    return penyakitLabels.indexOf(label);
}

// Helper function to get label from index
function getLabelFromIndex(index) {
    const penyakitLabels = dataPenyakit.map(p => p.Disease);
    return penyakitLabels[index];
}

function p(s) {
    return s.toLowerCase().trim();
}



