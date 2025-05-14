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
        <NeuralNetwork
          topLeftNeuronX={100}
          canvasHeight={canvasHeight}
          numLayers={5}
          neuronsPerLayer={[1, 3, 3, 3, 1]}
          neuronXGap={200}
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
