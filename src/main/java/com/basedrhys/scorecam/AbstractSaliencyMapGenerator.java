package com.basedrhys.scorecam;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.awt.image.BufferedImage;

// TODO document
public abstract class AbstractSaliencyMapGenerator {

    protected ComputationGraph computationGraph = null;

    protected int targetClassID = -1;

    protected int batchSize = 1;

    protected ImagePreProcessingScaler imagePreProcessingScaler = null;


    protected boolean imageChannelsLast = false;

    protected BufferedImage originalImage;

    protected BufferedImage heatmap;

    protected BufferedImage heatmapOnImage;

    public abstract void generateForImage(String inputImagePath);

    public ComputationGraph getComputationGraph() {
        return computationGraph;
    }

    public void setComputationGraph(ComputationGraph computationGraph) {
        this.computationGraph = computationGraph;
    }

    public int getTargetClassID() {
        return targetClassID;
    }

    public void setTargetClassID(int targetClassID) {
        this.targetClassID = targetClassID;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public ImagePreProcessingScaler getImagePreProcessingScaler() {
        return imagePreProcessingScaler;
    }

    public void setImagePreProcessingScaler(ImagePreProcessingScaler imagePreProcessingScaler) {
        this.imagePreProcessingScaler = imagePreProcessingScaler;
    }

    public BufferedImage getOriginalImage() {
        return originalImage;
    }

    public BufferedImage getHeatmap() {
        return heatmap;
    }

    public BufferedImage getHeatmapOnImage() {
        return heatmapOnImage;
    }

    public boolean isImageChannelsLast() {
        return imageChannelsLast;
    }

    public void setImageChannelsLast(boolean imageChannelsLast) {
        this.imageChannelsLast = imageChannelsLast;
    }
}
