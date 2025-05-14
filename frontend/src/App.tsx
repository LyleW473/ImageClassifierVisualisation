import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuralNetwork from './components/NeuralNetwork'

function App() {
  const neuronsPerLayer = [1, 3, 5, 5, 3, 1];
  const gapBetweenLayersX = 200;
  const neuronRadius = 50;
  const neuronSpacingY = 25;
  const neuronActiveColour = "green";
  const neuronInactiveColour = "red";

  // Calculate canvas height based on the number of neurons in the largest layer
  const canvasPaddingY = 0;
  const canvasPaddingX = 0;

  const maxNeurons = Math.max(...neuronsPerLayer);
  const totalNeuronHeight = maxNeurons * (2 * neuronRadius);
  const totalSpacing = (maxNeurons - 1) * neuronSpacingY;
  const layerHeight = totalNeuronHeight + totalSpacing;
  const canvasHeight = layerHeight + (2 * canvasPaddingY);
  
  // Calculate canvas width based on the number of layers and the gap between them
  const totalNetworkWidth = neuronsPerLayer.length * (2 * neuronRadius) + (neuronsPerLayer.length - 1) * gapBetweenLayersX;
  const canvasWidth = totalNetworkWidth + (2 * canvasPaddingX);

  return (
    <>
      <Heading title={"Hello World"} />
      <Section title={"Example Section"}>
          This is an example section
      </Section>
      <svg width={canvasWidth} height={canvasHeight}>

        // Background rectangle
        <rect
          x={0}
          y={0}
          width={canvasWidth}
          height={canvasHeight}
          fill="lightgrey"
        />

        // Neural network
        <NeuralNetwork
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          neuronsPerLayer={neuronsPerLayer}
          gapBetweenLayersX={gapBetweenLayersX}
          neuronRadius={neuronRadius}
          neuronSpacingY={neuronSpacingY}
          neuronActiveColour={neuronActiveColour}
          neuronInactiveColour={neuronInactiveColour}
        />
      </svg>
      <StartButton />
    </>
  )
}
export default App
