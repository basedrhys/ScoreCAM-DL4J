package com.basedrhys.scorecam;

import junit.framework.TestCase;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.net.URL;

@Log4j2
public class ScoreCAMTest extends TestCase {

    private static final String MODEL_HOME_URL = "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.2/";

    private static final String RESNET50_URL = MODEL_HOME_URL + "KerasResNet50.zip";

    private static final String RESNET101V2_URL = MODEL_HOME_URL + "KerasResNet101V2.zip";

    public ComputationGraph downloadModelFile(String remoteUrl) throws IOException {
        String localFilename = new File(remoteUrl).getName();

        File rootCacheDir = DL4JResources.getDirectory(ResourceType.ZOO_MODEL, "ScoreCAM");
        File cachedFile = new File(rootCacheDir, localFilename);

        if (!cachedFile.exists()) {
            log.info("Downloading model to " + cachedFile.toString());
            FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
        } else {
            log.info("Using cached model at " + cachedFile.toString());
        }

        return ModelSerializer.restoreComputationGraph(cachedFile);
    }

    public void testPretrainedModel(ScoreCAM scoreCAM, String modelName) {
        scoreCAM.setBatchSize(8);
        scoreCAM.generateForImage("src/test/resources/images/dog.jpg");

        try {
            ImageIO.write(scoreCAM.getOriginalImage(), "png", new File(String.format("output/%s_original.png", modelName)));
            ImageIO.write(scoreCAM.getHeatmap(), "png", new File(String.format("output/%s_heatmap.png", modelName)));
            ImageIO.write(scoreCAM.getHeatmapOnImage(), "png", new File(String.format("output/%s_heatmapOnImage.png", modelName)));
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void testGenerateForImageKerasResNet50() {
        try {
            ComputationGraph computationGraph = downloadModelFile(RESNET50_URL);

            ScoreCAM scoreCAM = new ScoreCAM();
            scoreCAM.setComputationGraph(computationGraph);

            testPretrainedModel(scoreCAM, "ResNet50");
        } catch (IOException ex) {
            ex.printStackTrace();
            fail();
        }
    }

    public void testGenerateForImageKerasResNet101V2() {
        try {
            ComputationGraph computationGraph = downloadModelFile(RESNET101V2_URL);

            ScoreCAM scoreCAM = new ScoreCAM();
            scoreCAM.setComputationGraph(computationGraph);
            scoreCAM.setImagePreProcessingScaler(new ImagePreProcessingScaler(-1, 1));

            testPretrainedModel(scoreCAM, "ResNet101V2");
        } catch (IOException ex) {
            ex.printStackTrace();
            fail();
        }
    }
}