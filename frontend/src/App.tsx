import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuralNetwork from './components/NeuralNetwork'

function App() {
  return (
    <>
      <Heading title={"Hello World"} />
      <Section title={"Example Section"}>
          This is an example section
      </Section>
      <svg width="2000" height="1000">
        <NeuralNetwork
          topLeftNeuronX={100}
          topLeftNeuronY={100}
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
