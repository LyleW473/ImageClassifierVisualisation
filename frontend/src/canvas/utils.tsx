const calculateCanvasWidth = (
                            neuronsPerLayer:number[], 
                            neuronRadius:number, 
                            gapBetweenLayersX:number, 
                            canvasPaddingX:number
                            ) => {
    /**
     * Calculates the width of the canvas based on the number of layers, the radius of the neurons, and the gap between layers.
     * @param neuronsPerLayer - An array representing the number of neurons in each layer.
     * @param neuronRadius - The radius of the neurons.
     * @param gapBetweenLayersX - The gap between layers in the x-direction.
     * @param canvasPaddingX - The padding on the left and right sides of the canvas.
     * @returns The total width of the canvas.
     */
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
    /**
     * Calculates the height of the canvas based on the number of neurons in the largest layer and the spacing between them.
     * @param neuronsPerLayer - An array representing the number of neurons in each layer.
     * @param neuronRadius - The radius of the neurons.
     * @param neuronSpacingY - The spacing between neurons in the y-direction.
     * @param canvasPaddingY - The padding on the top and bottom sides of the canvas.
     * @returns The total height of the canvas.
     */
    const maxNeurons = Math.max(...neuronsPerLayer);
    const totalNeuronHeight = maxNeurons * (2 * neuronRadius);
    const totalSpacing = (maxNeurons - 1) * neuronSpacingY;
    const layerHeight = totalNeuronHeight + totalSpacing;
    const canvasHeight = layerHeight + (2 * canvasPaddingY);
    return canvasHeight;
}

export { calculateCanvasWidth, calculateCanvasHeight };