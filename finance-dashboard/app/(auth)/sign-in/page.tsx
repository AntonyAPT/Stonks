import { Logo } from '../components'
import { SignInButton } from './SignInButton'

export default function SignInPage() {
  return (
    <>
      <Logo />
      <h1 className="text-2xl font-bold text-white mb-2">The Future of Personal Finance</h1>
      <p className="text-white/90 text-sm text-center max-w-xs mb-1 font-medium">
        AI-powered analysis of your finances
      </p>
      <p className="text-zinc-400 text-xs text-center max-w-xs mb-4">
        Uncover trends, keep track of your portfolio, and get personalized insights from your financial data.
      </p>
      <SignInButton />
      <p className="text-zinc-400 text-sm mt-4">
        Sign in with your Google account to get started
      </p>
    </>
  )
}