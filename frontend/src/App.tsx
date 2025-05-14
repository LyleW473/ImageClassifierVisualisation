import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuralNetwork from './components/NeuralNetwork'

function App() {
  const canvasWidth = 2000;
  const canvasHeight = 1000;
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
          numLayers={6}
          neuronsPerLayer={[1, 3, 5, 5, 3, 1]}
          layerXGap={200}
          neuronRadius={50}
          neuronSpacingY={25}
          neuronActiveColour="green"
          neuronInactiveColour="red"
        />
      </svg>
      <StartButton />
    </>
  )
}
export default App
