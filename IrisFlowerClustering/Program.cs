using Microsoft.ML;
using IrisFlowerClustering;

//Пути к файлам с входными и выходными данными
string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

//Создание контекста машинного обучения
var mlContext = new MLContext(seed: 0);

//Загрузка входных данных
IDataView dataView = mlContext.Data
    .LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

//Создание конвейера обучения
string featuresColumnName = "Features";
var pipeline = mlContext.Transforms
    .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
    .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

//Обучение модели
var model = pipeline.Fit(dataView);

//Сохранение модели
using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    mlContext.Model.Save(model, dataView.Schema, fileStream);
}

//Использование модели для прогнозирования
var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

//Тестовое предсказание
var prediction = predictor.Predict(TestIrisData.Setosa);
Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");

Console.WriteLine();
