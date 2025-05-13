import Heading from './components/Heading'
import Section from './components/Section'
import StartButton from './components/StartButton'

function App() {
  return (
    <>
      <Heading title={"Hello World"} />
      <Section title={"Example Section"}>
          This is an example section
      </Section>
      <StartButton />
    </>
  )
}

export default App
