import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuralNetwork from './components/NeuralNetwork'

function App() {
  const canvasWidth = 2000;
  const canvasHeight = 1000;
  const neuronsPerLayer = [1, 3, 5, 5, 3, 1];
  const gapBetweenLayersX = 200;
  const neuronRadius = 50;
  const neuronSpacingY = 25;
  const neuronActiveColour = "green";
  const neuronInactiveColour = "red";

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
