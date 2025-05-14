import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuralNetwork from './components/NeuralNetwork'
import { calculateCanvasHeight, calculateCanvasWidth } from './canvas/utils'

function App() {
  const neuronsPerLayer = [1, 3, 5, 5, 3, 1];
  const gapBetweenLayersX = 0;
  const neuronRadius = 50;
  const neuronSpacingY = 25;
  const neuronActiveColour = "green";
  const neuronInactiveColour = "red";

  // Calculate canvas height based on the number of neurons in the largest layer
  const canvasPaddingY = 0;
  const canvasPaddingX = 0;

  const canvasHeight = calculateCanvasHeight(neuronsPerLayer, neuronRadius, neuronSpacingY, canvasPaddingY);
  const canvasWidth = calculateCanvasWidth(neuronsPerLayer, neuronRadius, gapBetweenLayersX, canvasPaddingX);

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
