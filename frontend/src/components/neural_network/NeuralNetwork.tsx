import NeuronLayer from './NeuronLayer';
import Weight from './Weight';

type NeuralNetworkProperties = {
    canvasWidth: number;
    canvasHeight: number;
    neuronsPerLayer: number[];
    gapBetweenLayersX: number;
    neuronRadius: number;
    neuronPaddingX: number;
    neuronPaddingY: number;
    neuronSpacingY: number;
    neuronActiveColour?: string;
    neuronInactiveColour?: string;
}

const NeuralNetwork = ({
    canvasWidth,
    canvasHeight,
    neuronsPerLayer,
    gapBetweenLayersX,
    neuronRadius,
    neuronPaddingX,
    neuronPaddingY,
    neuronSpacingY,
    neuronActiveColour,
    neuronInactiveColour
    }: NeuralNetworkProperties) => {
    
    // Create the network layers
    const layers = [];
    const numLayers = neuronsPerLayer.length;
    const layerWidth = 2 * neuronRadius + 2 * neuronPaddingX;
    const totalNetworkWidth = numLayers * layerWidth + (numLayers - 1) * gapBetweenLayersX;
    const networkStart = (canvasWidth - totalNetworkWidth) / 2;

    for (let i = 0; i < numLayers; i++) {
        const numNeurons = neuronsPerLayer[i];

        // Center the layer vertically
        const layerHeight = numNeurons * (2 * neuronRadius) + (numNeurons - 1) * neuronSpacingY + 2 * neuronPaddingY;
        const layerYStart = (canvasHeight / 2) - (layerHeight / 2); 
        
        // Calculate the x position of the layer (ensuring that the layers are spaced out evenly throughout the canvas)
        const layerXStart = networkStart + (i * (layerWidth + gapBetweenLayersX));

        const neuralLayer = <NeuronLayer
            key={i}
            rectX={layerXStart}
            rectY={layerYStart}
            paddingX={neuronPaddingX}
            paddingY={neuronPaddingY}
            numNeurons={neuronsPerLayer[i]}
            neuronRadius={neuronRadius}
            neuronSpacingY={neuronSpacingY}
            neuronActiveColour={neuronActiveColour}
            neuronInactiveColour={neuronInactiveColour}
        />  
        layers.push(neuralLayer);
    }

    // Create the weights between neurons in adjacent layers
    const weights = [];
    for (let i = 0; i < numLayers - 1; i++) {
        const numNeuronsInLayer1 = neuronsPerLayer[i];
        const numNeuronsInLayer2 = neuronsPerLayer[i + 1];
        const layer1X = networkStart + (i * (layerWidth + gapBetweenLayersX)) + layerWidth / 2;
        const layer2X = networkStart + ((i + 1) * (layerWidth + gapBetweenLayersX)) + layerWidth / 2;

        const layer1Height = numNeuronsInLayer1 * (2 * neuronRadius) + (numNeuronsInLayer1 - 1) * neuronSpacingY + 2 * neuronPaddingY;
        const layer2Height = numNeuronsInLayer2 * (2 * neuronRadius) + (numNeuronsInLayer2 - 1) * neuronSpacingY + 2 * neuronPaddingY;
        const layer1YStart = (canvasHeight / 2) - (layer1Height / 2);
        const layer2YStart = (canvasHeight / 2) - (layer2Height / 2);

        for (let j = 0; j < numNeuronsInLayer1; j++) {
            const layer1Y = layer1YStart + (j * ((neuronRadius * 2) + neuronSpacingY)) + neuronRadius;
            for (let k = 0; k < numNeuronsInLayer2; k++) {
                const layer2Y = layer2YStart + (k * ((neuronRadius * 2) + neuronSpacingY)) + neuronRadius;
                weights.push(
                    <Weight
                        key={`${i}-${j}-${k}`}
                        x1={layer1X}
                        y1={layer1Y}
                        x2={layer2X}
                        y2={layer2Y}
                        isActive={false}
                    />
                );
            }
        }
    }
    return (
        <>
            {layers}
            {weights}
        </>
    );
}

export default NeuralNetwork;