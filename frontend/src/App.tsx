import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'
import NeuronLayer from './components/NeuronLayer'

function App() {
  return (
    <>
      <Heading title={"Hello World"} />
      <Section title={"Example Section"}>
          This is an example section
      </Section>
      <svg width="1000" height="1000">
        <NeuronLayer
          topLeftNeuronX={100}
          topLeftNeuronY={100}
          numNeurons={5}
          neuronRadius={50}
          neuronSpacing={20}
          neuronActiveColour="green"
          neuronInactiveColour="red"
          />
      </svg>
      <StartButton />
    </>
  )
}
export default App
