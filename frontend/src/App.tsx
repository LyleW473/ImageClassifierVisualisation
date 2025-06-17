import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuralNetwork from './components/neural_network/NeuralNetwork'
import { calculateCanvasHeight, calculateCanvasWidth } from './canvas/utils'

function App() {
  const neuronsPerLayer = [1, 3, 5, 5, 5, 3, 1];
  const gapBetweenLayersX = 200;
  const neuronRadius = 50;
  const neuronSpacingY = 25;
  const neuronPaddingX = 10;
  const neuronPaddingY = 10;
  const neuronActiveColour = "green";
  const neuronInactiveColour = "red";

  // Calculate canvas height based on the number of neurons in the largest layer
  const canvasPaddingY = 50;
  const canvasPaddingX = 50;

  const canvasHeight = calculateCanvasHeight(neuronsPerLayer, neuronRadius, neuronSpacingY, canvasPaddingY);
  const canvasWidth = calculateCanvasWidth(neuronsPerLayer, neuronRadius, gapBetweenLayersX, canvasPaddingX);

  return (
    <>
      <Heading title={"Hello World"} />
      <Section title={"Example Section"}>
          This is an example section
      </Section>
      <Section title={"Original Image"}>
          <img src="predicted_image_1.jpg" alt="Original" style={{ maxWidth: '100%' }} />
      </Section>
      <svg width={canvasWidth} height={canvasHeight}>

        // Background rectangle
        <rect
          x={0}
          y={0}
          width={canvasWidth}
          height={canvasHeight}
          fill="grey"
        />

        // Neural network
        <NeuralNetwork
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          neuronsPerLayer={neuronsPerLayer}
          gapBetweenLayersX={gapBetweenLayersX}
          neuronPaddingX={neuronPaddingX}
          neuronPaddingY={neuronPaddingY}
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
