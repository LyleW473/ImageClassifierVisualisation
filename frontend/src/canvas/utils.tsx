const calculateCanvasWidth = (
                            neuronsPerLayer:number[], 
                            neuronRadius:number, 
                            gapBetweenLayersX:number, 
                            canvasPaddingX:number
                            ) => {
    // Calculates the width of the canvas based on the number of layers and the gap between them, adding padding
    const totalNetworkWidth = neuronsPerLayer.length * (2 * neuronRadius) + (neuronsPerLayer.length - 1) * gapBetweenLayersX;
    const canvasWidth = totalNetworkWidth + (2 * canvasPaddingX);
    return canvasWidth;
}

const calculateCanvasHeight = (
                            neuronsPerLayer:number[], 
                            neuronRadius:number,
                            neuronSpacingY:number, 
                            canvasPaddingY:number
                            ) => {
    // Calculates the height of the canvas based on the number of neurons in the largest layer and the spacing between them, adding padding
    const maxNeurons = Math.max(...neuronsPerLayer);
    const totalNeuronHeight = maxNeurons * (2 * neuronRadius);
    const totalSpacing = (maxNeurons - 1) * neuronSpacingY;
    const layerHeight = totalNeuronHeight + totalSpacing;
    const canvasHeight = layerHeight + (2 * canvasPaddingY);
    return canvasHeight;
}

export { calculateCanvasWidth, calculateCanvasHeight };